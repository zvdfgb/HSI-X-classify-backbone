import torch
import torch.nn as nn
import torch.nn.functional as F

from .dct import DCT2D


class SMCB(nn.Module):
    """
    Semantic Mixing Convolution Block (SMCB)
    Generates semantic-aware dynamic convolution kernels according to contextual relationships.
    Reference: SGFNet (Zhao et al., "Semantic-Guided Fusion Network for
    Multi-source Remote Sensing Image Classification", IEEE GRSL).
    """
    def __init__(self, channels, kernel_size=7, context_grid=3):
        super(SMCB, self).__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.context_grid = context_grid
        self.k2 = kernel_size * kernel_size
        self.s2 = context_grid * context_grid

        # 1. Feature projections
        self.proj_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.proj_q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # 2. Compact context extraction
        self.pool_k = nn.AdaptiveAvgPool2d(context_grid)
        self.proj_k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # 3. Dynamic kernel generator: transforms context affinity to KxK kernel weights
        self.kernel_generator = nn.Sequential(
            nn.Linear(self.s2, self.k2, bias=True),
            nn.Softmax(dim=-1),
        )
        
        # 4. Local enhancement branch (Depthwise Convolution)
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
        self.dyn_bn = nn.BatchNorm2d(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Projections
        v = self.proj_v(x)  # [B, C, H, W]
        q = self.proj_q(x)  # [B, C, H, W]
        
        # Compact context
        k = self.proj_k(self.pool_k(x))  # [B, C, S, S]
        
        # Flatten spatial dimensions for affinity calculation
        q_flat = q.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        k_flat = k.view(B, C, self.s2)                  # [B, C, S^2]
        
        # Affinity matrix A = Q^T * K / sqrt(C)
        affinity = torch.bmm(q_flat, k_flat) * (1.0 / (C ** 0.5))  # [B, HW, S^2]
        
        # Generate position-specific dynamic kernels
        # D: [B, HW, K^2] -> [B, H, W, K^2]
        dyn_kernel = self.kernel_generator(affinity).view(B, H, W, self.k2)
        
        # Unfold value feature map V to extract local patches
        # V_unfold: [B, C * K^2, HW] -> [B, C, K^2, H, W] -> [B, H, W, C, K^2]
        v_unfold = F.unfold(v, kernel_size=self.kernel_size, padding=self.padding)
        v_unfold = v_unfold.view(B, C, self.k2, H, W).permute(0, 3, 4, 1, 2)
        
        # Dynamic convolution: sum over local kernel elements
        # [B, H, W, 1, K^2] * [B, H, W, C, K^2] -> sum over dim=-1 -> [B, H, W, C]
        dyn_out = torch.sum(v_unfold * dyn_kernel.unsqueeze(3), dim=-1)
        dyn_out = dyn_out.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        dyn_out = self.dyn_bn(dyn_out)
        
        # Local enhancement branch
        local_out = self.local_conv(x)
        
        # Residual combination
        out = self.act(x + dyn_out + local_out)
        return out


class FMFB(nn.Module):
    """
    Frequency Modulated Fusion Block (FMFB)
    Performs cross-modal interaction in the frequency domain using 2D-DCT/IDCT
    to alleviate spatial misalignment and enhance complementary features.
    """
    def __init__(self, channels, height=11, width=11):
        super(FMFB, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
        # 2D DCT module
        self.dct = DCT2D(height=height, width=width)
        
        # Frequency domain cross-modal projections
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.vh_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.vx_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        
        # Feedforward networks in spatial domain after IDCT
        self.ffn_h = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.ffn_x = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, fh, fx):
        B, C, H, W = fh.shape
        
        # 1. 2D-DCT to frequency domain
        f_hat_h = self.dct(fh)  # [B, C, H, W]
        f_hat_x = self.dct(fx)  # [B, C, H, W]
        
        # 2. Frequency difference
        f_hat_diff = f_hat_h - f_hat_x  # [B, C, H, W]
        
        # 3. Cross-modal frequency attention projections
        qd = self.q_proj(f_hat_diff)  # [B, C, H, W]
        kd = self.k_proj(f_hat_diff)  # [B, C, H, W]
        vh = self.vh_proj(f_hat_h)    # [B, C, H, W]
        vx = self.vx_proj(f_hat_x)    # [B, C, H, W]
        
        # Flatten spatial frequency grid
        qd_flat = qd.view(B, C, H * W).permute(0, 2, 1)  # [B, HW, C]
        kd_flat = kd.view(B, C, H * W)                  # [B, C, HW]
        
        # Frequency attention matrix
        attn_freq = torch.bmm(qd_flat, kd_flat) * (1.0 / (C ** 0.5))
        attn_freq = F.softmax(attn_freq, dim=-1)  # [B, HW, HW]
        
        vh_flat = vh.view(B, C, H * W)  # [B, C, HW]
        vx_flat = vx.view(B, C, H * W)  # [B, C, HW]
        
        # Modulate frequency features
        f_attn_h = torch.bmm(vh_flat, attn_freq.transpose(1, 2)).view(B, C, H, W)
        f_attn_x = torch.bmm(vx_flat, attn_freq.transpose(1, 2)).view(B, C, H, W)
        
        # 4. Inverse 2D-DCT back to spatial domain
        f_spatial_h = self.dct.inverse(f_attn_h)
        f_spatial_x = self.dct.inverse(f_attn_x)
        
        # 5. Spatial FFN & residual connection
        fh_out = self.act(fh + self.ffn_h(f_spatial_h))
        fx_out = self.act(fx + self.ffn_x(f_spatial_x))
        
        return fh_out, fx_out


class SGFM(nn.Module):
    """
    Semantic-Guided Fusion Module (SGFM)
    Combines SMCB and FMFB in a hierarchical refinement structure.
    """
    def __init__(self, channels, height=11, width=11, kernel_size=7):
        super(SGFM, self).__init__()
        self.smcb_h1 = SMCB(channels, kernel_size=kernel_size)
        self.smcb_x1 = SMCB(channels, kernel_size=kernel_size)
        self.fmfb = FMFB(channels, height=height, width=width)
        self.smcb_h2 = SMCB(channels, kernel_size=kernel_size)
        self.smcb_x2 = SMCB(channels, kernel_size=kernel_size)

    def forward(self, fh, fx):
        # Initial semantic feature refinement
        fh = self.smcb_h1(fh)
        fx = self.smcb_x1(fx)
        
        # Cross-modal frequency modulated fusion
        fh, fx = self.fmfb(fh, fx)
        
        # Deep semantic refinement
        fh = self.smcb_h2(fh)
        fx = self.smcb_x2(fx)
        
        return fh, fx


class SGFNet(nn.Module):
    """
    Semantic-Guided Fusion Network (SGFNet)
    Reference: Zhao et al., "Semantic-Guided Fusion Network for Multi-source
    Remote Sensing Image Classification", IEEE GRSL.
    """
    def __init__(self, hsi_channels=30, aux_channels=1, num_classes=7,
                 hidden_dim=64, num_sgfm=2, window_size=11, smcb_kernel_size=7):
        super(SGFNet, self).__init__()
        self.hsi_channels = hsi_channels
        self.aux_channels = aux_channels
        self.num_classes = num_classes
        self.window_size = window_size
        
        # 1. HSI 3D Convolution Preprocessing
        # Input: [B, 1, C_pca, H, W]
        self.hsi_conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.hsi_proj2d = nn.Sequential(
            nn.Conv2d(8 * hsi_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 2. SAR/LiDAR 2D Convolution Preprocessing
        # Input: [B, C_aux, H, W]
        self.aux_conv2d = nn.Sequential(
            nn.Conv2d(aux_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 3. Stacked SGFM Modules (default: 2)
        self.sgfms = nn.ModuleList([
            SGFM(hidden_dim, height=window_size, width=window_size, kernel_size=smcb_kernel_size)
            for _ in range(num_sgfm)
        ])
        
        # 4. Refinement and Classifier Head
        self.refine_conv = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, hsi_pca, aux):
        """
        hsi_pca: [B, C_pca, H, W] or [B, 1, C_pca, H, W]
        aux: [B, C_aux, H, W]
        """
        if hsi_pca.dim() == 4:
            hsi_pca = hsi_pca.unsqueeze(1)  # [B, 1, C_pca, H, W]
            
        B, _, C, H, W = hsi_pca.shape
        
        # HSI preprocessing: 3D conv -> 2D projection
        feat_h = self.hsi_conv3d(hsi_pca)
        feat_h = feat_h.view(B, 8 * C, H, W)
        fh = self.hsi_proj2d(feat_h)
        
        # SAR/LiDAR preprocessing: 2D conv
        fx = self.aux_conv2d(aux)
        
        # Stacked SGFMs
        for sgfm in self.sgfms:
            fh, fx = sgfm(fh, fx)
            
        # Refinement & Classification
        f_fused = torch.cat([fh, fx], dim=1)
        f_refined = self.refine_conv(f_fused)
        f_pooled = self.gap(f_refined).view(B, -1)
        logits = self.classifier(f_pooled)
        
        return logits
