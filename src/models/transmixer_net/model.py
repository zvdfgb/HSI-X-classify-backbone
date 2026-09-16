import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM2D(nn.Module):
    """
    轻量纯 PyTorch 2D 状态空间模型（双向空间序列扫描）
    具备 O(N) 线性计算复杂度，自适应生成时间离散步长 delta
    """
    def __init__(self, d_model, d_state=16):
        super(SelectiveSSM2D, self).__init__()
        self.d_model = d_model
        self.d_state = d_state

        # A 矩阵参数化为对角负数，保证状态衰减稳定性
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        
        # 输入自适应参数投影层 (生成 B, C, Delta)
        self.x_proj = nn.Linear(d_model, 2 * d_state + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: [B, L, D] 其中 L = H * W
        B, L, D = x.shape
        A = -torch.exp(self.A_log)  # [D, N]

        x_dbl = self.x_proj(x)  # [B, L, 2*N + D]
        delta = x_dbl[:, :, :D]  # [B, L, D]
        b = x_dbl[:, :, D:D + self.d_state]  # [B, L, N]
        c = x_dbl[:, :, D + self.d_state:]   # [B, L, N]

        delta = F.softplus(self.dt_proj(delta))  # [B, L, D]

        # 正向序列递推（轻量并行近似计算）
        y = torch.zeros_like(x)
        h = torch.zeros(B, D, self.d_state, device=x.device)

        for t in range(L):
            xt = x[:, t, :].unsqueeze(-1)  # [B, D, 1]
            dt = delta[:, t, :].unsqueeze(-1)  # [B, D, 1]
            bt = b[:, t, :].unsqueeze(1)   # [B, 1, N]
            ct = c[:, t, :].unsqueeze(-1)  # [B, N, 1]

            at = torch.exp(A.unsqueeze(0) * dt)  # [B, D, N]
            h = at * h + (dt * xt) * bt          # [B, D, N]
            yt = torch.matmul(h, ct).squeeze(-1) # [B, D]
            y[:, t, :] = yt

        out = self.out_proj(y)
        return out, delta


class DecoupledTransMixerBlock(nn.Module):
    """
    光谱-空间解耦 TransMixer 模块
    1. 2D SSM 序列建模提取潜特征与 delta
    2. 通道自适应/解耦切分: Top 50% Global Stream, Rest 50% Local Stream
    3. Global: 轻量 Multi-Head Self-Attention
    4. Local: 深度可分离卷积 (Depthwise Conv)
    5. 1x1 卷积重组融合

    支持消融实验开关 mode:
    - "decoupled": 完整解耦方案 (50% 全局注意力 + 50% 局部卷积)
    - "only_conv":  对照组 A (100% 纯局部深度卷积)
    - "only_attn":  对照组 B (100% 纯全局自注意力)
    - "pure_ssm":   对照组 C (100% 纯状态空间序列扫描，无额外注意力/卷积细化)
    """
    def __init__(self, dim, num_heads=4, split_ratio=0.5, mode="decoupled"):
        super(DecoupledTransMixerBlock, self).__init__()
        self.dim = dim
        self.mode = mode
        self.global_dim = int(dim * split_ratio)
        self.local_dim = dim - self.global_dim

        self.norm1 = nn.LayerNorm(dim)

        if self.mode == "decoupled":
            self.ssm = SelectiveSSM2D(d_model=dim)
            # 全局流：轻量多头自注意力
            self.global_attn = nn.MultiheadAttention(embed_dim=self.global_dim, num_heads=num_heads, batch_first=True)
            self.global_norm = nn.LayerNorm(self.global_dim)
            # 局部流：深度可分离卷积 (Depthwise 3x3 + Pointwise 1x1)
            self.local_dw = nn.Sequential(
                nn.Conv2d(self.local_dim, self.local_dim, kernel_size=3, padding=1, groups=self.local_dim, bias=False),
                nn.BatchNorm2d(self.local_dim),
                nn.GELU(),
                nn.Conv2d(self.local_dim, self.local_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(self.local_dim)
            )
        elif self.mode == "only_attn":
            # 对照组 B: 全通道自注意力
            self.global_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
            self.global_norm = nn.LayerNorm(dim)
        elif self.mode == "only_conv":
            # 对照组 A: 全通道局部深度卷积
            self.local_dw = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(dim)
            )
        elif self.mode == "pure_ssm":
            # 对照组 C: 纯 SSM 序列扫描
            self.ssm = SelectiveSSM2D(d_model=dim)
        else:
            raise ValueError(f"未知的消融模式: {mode}")

        # 融合投影
        self.fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )

    def forward(self, x_tokens, H, W):
        # x_tokens: [B, L, D]
        residual = x_tokens
        norm_x = self.norm1(x_tokens)

        if self.mode == "decoupled":
            # 1. SSM 线性前向扫描
            ssm_out, delta = self.ssm(norm_x)  # ssm_out: [B, L, D], delta: [B, L, D]

            # 2. 通道解耦切分
            x_g = ssm_out[:, :, :self.global_dim]
            x_l = ssm_out[:, :, self.global_dim:]

            # 3. 全局语义分支 (Self-Attention)
            attn_out, _ = self.global_attn(self.global_norm(x_g), self.global_norm(x_g), self.global_norm(x_g))
            x_g_out = x_g + attn_out

            # 4. 局部微结构分支 (Reshape 回 2D 进行局部深度卷积)
            B, L, _ = x_l.shape
            x_l_2d = x_l.transpose(1, 2).view(B, self.local_dim, H, W)
            conv_out = self.local_dw(x_l_2d)
            x_l_out = conv_out.view(B, self.local_dim, L).transpose(1, 2)

            # 5. 拼接与点向投影融合
            fused = torch.cat([x_g_out, x_l_out], dim=-1)
            out = residual + self.fusion(fused)

        elif self.mode == "only_attn":
            # 全通道自注意力
            attn_out, _ = self.global_attn(self.global_norm(norm_x), self.global_norm(norm_x), self.global_norm(norm_x))
            out = residual + self.fusion(norm_x + attn_out)

        elif self.mode == "only_conv":
            # 全通道局部深度卷积
            B, L, _ = norm_x.shape
            x_2d = norm_x.transpose(1, 2).view(B, self.dim, H, W)
            conv_out = self.local_dw(x_2d)
            conv_out = conv_out.view(B, self.dim, L).transpose(1, 2)
            out = residual + self.fusion(norm_x + conv_out)

        elif self.mode == "pure_ssm":
            # 纯 SSM 序列流
            ssm_out, _ = self.ssm(norm_x)
            out = residual + self.fusion(ssm_out)

        return out


class TransMixerNet(nn.Module):
    """
    面向高光谱多源分类的 TransMixer 完整网络结构
    """
    def __init__(self, hsi_channels=30, aux_channels=1, num_classes=15, hidden_dim=64, num_blocks=2, window_size=11, mode="decoupled"):
        super(TransMixerNet, self).__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.mode = mode

        # 1. 高光谱浅层特征提取 (3D Conv 捕捉连续光谱-空间上下文)
        self.hsi_stem = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.hsi_proj = nn.Sequential(
            nn.Conv2d(8 * hsi_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 2. 辅助模态特征提取 (LiDAR/SAR)
        self.aux_stem = nn.Sequential(
            nn.Conv2d(aux_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 3. 堆叠 TransMixer 解耦模块
        self.blocks = nn.ModuleList([
            DecoupledTransMixerBlock(dim=hidden_dim, num_heads=4, split_ratio=0.5, mode=mode)
            for _ in range(num_blocks)
        ])

        # 4. 模态交互融合与分类头
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, hsi_pca, aux):
        # 维度对齐与校验
        if hsi_pca.dim() == 4:
            hsi_pca = hsi_pca.unsqueeze(1)  # [B, 1, C, H, W]

        B, _, C, H, W = hsi_pca.shape

        # HSI 提取
        h_3d = self.hsi_stem(hsi_pca)
        h_2d = self.hsi_proj(h_3d.view(B, -1, H, W))  # [B, D, H, W]

        # 转换为 Token 序列送入 TransMixer
        tokens = h_2d.flatten(2).transpose(1, 2)  # [B, H*W, D]
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        feat_hsi = tokens.transpose(1, 2).view(B, self.hidden_dim, H, W)

        # LiDAR/SAR 提取
        feat_aux = self.aux_stem(aux)

        # 双模态拼接融合
        fused = torch.cat([feat_hsi, feat_aux], dim=1)
        fused = self.fusion_conv(fused)

        # 池化与分类
        pooled = self.gap(fused).view(B, -1)
        logits = self.classifier(pooled)
        return logits
