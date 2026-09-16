import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionGuidedEdgeGatedConv(nn.Module):
    """
    方向引导边缘门控卷积 (DEGConv - Direction-guided Edge Gated Convolution)
    源自 MixerCSeg 核心创新模块：
    1. 计算方向梯度嵌入 (Sobel 算子抽取水平/垂直方向特征与夹角)
    2. 条形卷积 (Strip Convolution 1xk & kx1) 专攻各向异性细长几何结构 (如道路、铁路、跑道)
    3. 动态方向门控加权 (Direction Gating)
    """
    def __init__(self, in_channels, strip_kernel_size=5):
        super(DirectionGuidedEdgeGatedConv, self).__init__()
        self.in_channels = in_channels
        k = strip_kernel_size
        pad = k // 2

        # 1. 条形卷积 (Strip Convolutions)
        # 横向 1xk + 纵向 kx1 并行提取延伸线状特征
        self.strip_h = nn.Conv2d(in_channels, in_channels, kernel_size=(1, k), padding=(0, pad), groups=in_channels, bias=False)
        self.strip_v = nn.Conv2d(in_channels, in_channels, kernel_size=(k, 1), padding=(pad, 0), groups=in_channels, bias=False)
        self.strip_proj = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.GELU()
        )

        # 2. 方向梯度先验提取层 (Sobel 算子 -> 动态门控权重)
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 8.0
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

        # 梯度特征融合与门控网络
        self.gate_conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 1. 计算通道均值灰度图以提取空间几何物理边缘 (梯度截断保护，避免平坦区 atan2(0,0) 反向传播产生 NaN)
        with torch.no_grad():
            x_mean = torch.mean(x.detach(), dim=1, keepdim=True)  # [B, 1, H, W]
            dx = F.conv2d(x_mean, self.sobel_x, padding=1)
            dy = F.conv2d(x_mean, self.sobel_y, padding=1)
            mag = torch.sqrt(dx ** 2 + dy ** 2 + 1e-6)
            angle = torch.atan2(dy, dx)
            grad_feat = torch.cat([mag, angle], dim=1)  # [B, 2, H, W]

        # 2. 动态方向门控加权
        gate = self.gate_conv(grad_feat).view(B, C, 1, 1)  # [B, C, 1, 1]

        # 3. 条形卷积并联提取
        feat_h = self.strip_h(x)
        feat_v = self.strip_v(x)
        strip_cat = torch.cat([feat_h, feat_v], dim=1)
        strip_out = self.strip_proj(strip_cat)

        # 4. 门控调制与残差融合
        out = x + strip_out * gate
        return out


class SelectiveSSM2D(nn.Module):
    """
    轻量纯 PyTorch 2D 状态空间模型（双向空间序列扫描）
    与 TransMixerNet 保持 100% 完全一致
    """
    def __init__(self, d_model, d_state=16):
        super(SelectiveSSM2D, self).__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_model, 1)))
        self.x_proj = nn.Linear(d_model, 2 * d_state + d_model, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, D = x.shape
        A = -torch.exp(self.A_log)

        x_dbl = self.x_proj(x)
        delta = x_dbl[:, :, :D]
        b = x_dbl[:, :, D:D + self.d_state]
        c = x_dbl[:, :, D + self.d_state:]

        delta = F.softplus(self.dt_proj(delta))

        y = torch.zeros_like(x)
        h = torch.zeros(B, D, self.d_state, device=x.device)

        for t in range(L):
            xt = x[:, t, :].unsqueeze(-1)
            dt = delta[:, t, :].unsqueeze(-1)
            bt = b[:, t, :].unsqueeze(1)
            ct = c[:, t, :].unsqueeze(-1)

            at = torch.exp(A.unsqueeze(0) * dt)
            h = at * h + (dt * xt) * bt
            yt = torch.matmul(h, ct).squeeze(-1)
            y[:, t, :] = yt

        out = self.out_proj(y)
        return out, delta


class TransMixerDEGBlock(nn.Module):
    """
    消融/融合构建块：
    1. arch == 'transmixer_deg' (A + B 方案):
       SSM 扫描 -> 前 50% 走全局自注意力，后 50% 走 DEGConv 几何边缘门控卷积 -> Concat 融合
    2. arch == 'deg_only' (单独 B 方案):
       100% 纯 DEGConv 各向异性条形门控提取，不带通道解耦与注意力
    """
    def __init__(self, dim, num_heads=4, split_ratio=0.5, arch="transmixer_deg"):
        super(TransMixerDEGBlock, self).__init__()
        self.dim = dim
        self.arch = arch
        self.global_dim = int(dim * split_ratio)
        self.local_dim = dim - self.global_dim

        self.norm1 = nn.LayerNorm(dim)

        if self.arch == "transmixer_deg":
            self.ssm = SelectiveSSM2D(d_model=dim)
            # 全局流：轻量多头自注意力
            self.global_attn = nn.MultiheadAttention(embed_dim=self.global_dim, num_heads=num_heads, batch_first=True)
            self.global_norm = nn.LayerNorm(self.global_dim)
            # 局部流：升级为 DEGConv (条形卷积 + 方向门控)
            self.local_deg = DirectionGuidedEdgeGatedConv(in_channels=self.local_dim, strip_kernel_size=5)
        elif self.arch == "deg_only":
            # 单独 B：全通道直接接 DEGConv 提取
            self.deg_conv = DirectionGuidedEdgeGatedConv(in_channels=dim, strip_kernel_size=5)
        else:
            raise ValueError(f"未知的架构选项: {arch}")

        self.fusion = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU()
        )

    def forward(self, x_tokens, H, W):
        residual = x_tokens
        norm_x = self.norm1(x_tokens)

        if self.arch == "transmixer_deg":
            # 1. SSM 序列扫描
            ssm_out, _ = self.ssm(norm_x)

            # 2. 通道解耦切分
            x_g = ssm_out[:, :, :self.global_dim]
            x_l = ssm_out[:, :, self.global_dim:]

            # 3. 全局语义流 (Self-Attention)
            attn_out, _ = self.global_attn(self.global_norm(x_g), self.global_norm(x_g), self.global_norm(x_g))
            x_g_out = x_g + attn_out

            # 4. 局部几何流 (DEGConv: 条形卷积 + 方向门控)
            B, L, _ = x_l.shape
            x_l_2d = x_l.transpose(1, 2).view(B, self.local_dim, H, W)
            deg_out_2d = self.local_deg(x_l_2d)
            x_l_out = deg_out_2d.view(B, self.local_dim, L).transpose(1, 2)

            # 5. 拼接融合
            fused = torch.cat([x_g_out, x_l_out], dim=-1)
            out = residual + self.fusion(fused)

        elif self.arch == "deg_only":
            # 单独 B: 100% 走 DEGConv
            B, L, _ = norm_x.shape
            x_2d = norm_x.transpose(1, 2).view(B, self.dim, H, W)
            deg_out_2d = self.deg_conv(x_2d)
            deg_out = deg_out_2d.view(B, self.dim, L).transpose(1, 2)
            out = residual + self.fusion(norm_x + deg_out)

        return out


class TransMixerDEGNet(nn.Module):
    """
    高光谱多源分类主干网络（支持 单独 B 与 A+B 融合）
    严格控制变量：除了特征提取块外，Stem、Fusion 与分类头与 TransMixerNet 100% 完全一致
    """
    def __init__(self, hsi_channels=30, aux_channels=1, num_classes=15, hidden_dim=64, num_blocks=2, window_size=11, arch="transmixer_deg"):
        super(TransMixerDEGNet, self).__init__()
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.arch = arch

        # 1. 高光谱浅层特征提取 (与 TransMixerNet 100% 相同)
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

        # 2. 辅助模态特征提取 (与 TransMixerNet 100% 相同)
        self.aux_stem = nn.Sequential(
            nn.Conv2d(aux_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )

        # 3. 堆叠特征提取块 (核心创新模块，支持 deg_only 与 transmixer_deg)
        self.blocks = nn.ModuleList([
            TransMixerDEGBlock(dim=hidden_dim, num_heads=4, split_ratio=0.5, arch=arch)
            for _ in range(num_blocks)
        ])

        # 4. 模态交互融合与分类头 (与 TransMixerNet 100% 相同)
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
        if hsi_pca.dim() == 4:
            hsi_pca = hsi_pca.unsqueeze(1)  # [B, 1, C, H, W]

        B, _, C, H, W = hsi_pca.shape

        # HSI 浅层提取
        h_3d = self.hsi_stem(hsi_pca)
        h_2d = self.hsi_proj(h_3d.view(B, -1, H, W))  # [B, D, H, W]

        # Token 序列特征提取
        tokens = h_2d.flatten(2).transpose(1, 2)  # [B, H*W, D]
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        feat_hsi = tokens.transpose(1, 2).view(B, self.hidden_dim, H, W)

        # LiDAR/SAR 提取
        feat_aux = self.aux_stem(aux)

        # 双模态拼接与融合
        fused = torch.cat([feat_hsi, feat_aux], dim=1)
        fused = self.fusion_conv(fused)

        # 分类预测
        pooled = self.gap(fused).view(B, -1)
        logits = self.classifier(pooled)
        return logits

