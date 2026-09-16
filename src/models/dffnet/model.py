import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention for HSI spectral feature enhancement
    F'_h = F_h + C1x1(sigma(C1x1(GAP(F_h))))
    """
    def __init__(self, channels, reduction=4):
        super(ChannelAttention, self).__init__()
        mid_channels = max(channels // reduction, 8)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, channels, kernel_size=1, bias=False),
        )

    def forward(self, x):
        att = self.fc(self.gap(x))
        return x + att


class SpatialAttention(nn.Module):
    """
    Spatial Attention for SAR/LiDAR spatial/structural feature enhancement
    F'_x = F_x + C5x5([GAP(F_x), GMP(F_x)])
    """
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_pool = torch.cat([avg_out, max_out], dim=1)
        att = self.conv(spatial_pool)
        return x + (x * att)


def channel_shuffle(x, groups=2):
    """
    Channel shuffle operation for interleaving channels of HSI and SAR/LiDAR
    """
    batch_size, num_channels, height, width = x.shape
    channels_per_group = num_channels // groups
    
    # Reshape: [B, groups, channels_per_group, H, W]
    x = x.view(batch_size, groups, channels_per_group, height, width)
    # Transpose: [B, channels_per_group, groups, H, W]
    x = torch.transpose(x, 1, 2).contiguous()
    # Flatten back to [B, num_channels, H, W]
    x = x.view(batch_size, -1, height, width)
    return x


class SSAFB(nn.Module):
    """
    Spectral-Spatial Adaptive Fusion Block (SSAFB)
    F_o = C1x1(CS([F'_h, F'_x]))
    """
    def __init__(self, channels):
        super(SSAFB, self).__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention(kernel_size=5)
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, fh, fx):
        # 1. Attention enhancement on respective modalities
        fh_prime = self.channel_att(fh)
        fx_prime = self.spatial_att(fx)
        
        # 2. Concatenate along channel dimension
        f_cat = torch.cat([fh_prime, fx_prime], dim=1)
        
        # 3. Channel shuffle
        f_shuffled = channel_shuffle(f_cat, groups=2)
        
        # 4. Dimension reduction
        fo = self.fuse_conv(f_shuffled)
        return fo


class FrequencyFeedForward(nn.Module):
    """
    Feedforward Network (FFN) with Identity, Spatial Conv, and Frequency Conv branches
    """
    def __init__(self, channels):
        super(FrequencyFeedForward, self).__init__()
        # Spatial Depthwise Conv Branch
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        
        # Frequency Depthwise Conv Branch (processes real and imag components)
        self.freq_conv_real = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.freq_conv_imag = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.freq_bn = nn.BatchNorm2d(channels)
        self.freq_act = nn.ReLU(inplace=True)

    def forward(self, x):
        # Identity
        identity = x
        
        # Spatial branch
        spatial_out = self.spatial_conv(x)
        
        # Frequency branch
        B, C, H, W = x.shape
        fft_feat = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        real_part = self.freq_conv_real(fft_feat.real)
        imag_part = self.freq_conv_imag(fft_feat.imag)
        freq_feat = torch.complex(real_part, imag_part)
        freq_spatial = torch.fft.irfft2(freq_feat, s=(H, W), dim=(-2, -1), norm="ortho")
        freq_out = self.freq_act(self.freq_bn(freq_spatial))
        
        return identity + spatial_out + freq_out


class DFB(nn.Module):
    """
    Dynamic Filter Block (DFB)
    Generates dynamic frequency-domain filter K(X_in) using learnable filter bases F = {F_1, ..., F_N}
    and applies elementwise multiplication in frequency domain.
    """
    def __init__(self, channels, height=11, width=11, num_basis=4):
        super(DFB, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.num_basis = num_basis
        
        freq_w = width // 2 + 1
        # Learnable filter bases (real and imag components)
        self.filter_basis_real = nn.Parameter(torch.randn(num_basis, channels, height, freq_w) * 0.02)
        self.filter_basis_imag = nn.Parameter(torch.randn(num_basis, channels, height, freq_w) * 0.02)
        
        # Dynamic weights generation from input features
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, max(channels // 4, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // 4, 8), num_basis),
            nn.Softmax(dim=-1),
        )
        
        # FFN module
        self.ffn = FrequencyFeedForward(channels)

    def forward(self, x):
        B, C, H, W = x.shape
        freq_w = W // 2 + 1
        
        # 1. 2D Real FFT to frequency domain
        f_uv = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        
        # 2. Dynamic weights generation via GAP + MLP
        pooled = self.gap(x).view(B, C)
        weights = self.mlp(pooled)  # [B, num_basis]
        
        # 3. Dynamic filter kernel aggregation
        w_expanded = weights.view(B, self.num_basis, 1, 1, 1)
        basis_r = self.filter_basis_real.unsqueeze(0)  # [1, N, C, H, W_freq]
        basis_i = self.filter_basis_imag.unsqueeze(0)  # [1, N, C, H, W_freq]
        
        k_real = torch.sum(w_expanded * basis_r, dim=1)
        k_imag = torch.sum(w_expanded * basis_i, dim=1)
        k_dynamic = torch.complex(k_real, k_imag)
        
        # 4. Frequency domain dynamic filtering: \hat{f}(u, v) = K(X_in) \odot f(u, v)
        f_hat = f_uv * (1.0 + k_dynamic)  # residual connection in frequency domain
        
        # 5. Inverse FFT back to spatial domain
        x_out = torch.fft.irfft2(f_hat, s=(H, W), dim=(-2, -1), norm="ortho")
        
        # 6. Nonlinear transformation via FFN
        out = self.ffn(x_out)
        return out


class DFFM(nn.Module):
    """
    Dynamic Filter Fusion Module (DFFM)
    Combines SSAFB and DFB for cross-modal feature extraction and fusion
    """
    def __init__(self, channels, height=11, width=11, num_basis=4):
        super(DFFM, self).__init__()
        self.ssafb = SSAFB(channels)
        self.dfb_h = DFB(channels, height=height, width=width, num_basis=num_basis)
        self.dfb_x = DFB(channels, height=height, width=width, num_basis=num_basis)

    def forward(self, fh, fx):
        # 1. SSAFB cross-modal fusion
        fo = self.ssafb(fh, fx)
        
        # 2. Residual summation
        fh_res = fh + fo
        fx_res = fx + fo
        
        # 3. Dynamic frequency filtering on each branch
        fh_out = self.dfb_h(fh_res)
        fx_out = self.dfb_x(fx_res)
        
        return fh_out, fx_out


class DFFNet(nn.Module):
    """
    Dynamic Frequency Feature Fusion Network (DFFNet)
    Reference: Zhao et al., "Dynamic Frequency Feature Fusion Network for
    Multisource Remote Sensing Data Classification", IEEE GRSL 2025.
    """
    def __init__(self, hsi_channels=30, aux_channels=1, num_classes=15,
                 hidden_dim=64, num_dffm=2, window_size=11, num_basis=4):
        super(DFFNet, self).__init__()
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
        
        # 3. Stacked DFFM Modules (default: 2)
        self.dffms = nn.ModuleList([
            DFFM(hidden_dim, height=window_size, width=window_size, num_basis=num_basis)
            for _ in range(num_dffm)
        ])
        
        # 4. Refinement and Classifier Module
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
        feat_h = self.hsi_conv3d(hsi_pca)  # [B, 8, C, H, W]
        feat_h = feat_h.view(B, 8 * C, H, W)
        fh = self.hsi_proj2d(feat_h)  # [B, hidden_dim, H, W]
        
        # Auxiliary (SAR/LiDAR) preprocessing: 2D conv
        fx = self.aux_conv2d(aux)  # [B, hidden_dim, H, W]
        
        # Stacked DFFMs
        for dffm in self.dffms:
            fh, fx = dffm(fh, fx)
            
        # Refinement & Classification
        f_fused = torch.cat([fh, fx], dim=1)  # [B, 2 * hidden_dim, H, W]
        f_refined = self.refine_conv(f_fused)  # [B, hidden_dim, H, W]
        f_pooled = self.gap(f_refined).view(B, -1)  # [B, hidden_dim]
        logits = self.classifier(f_pooled)  # [B, num_classes]
        
        return logits
