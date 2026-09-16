import torch
import torch.nn as nn
import numpy as np


class DCT2D(nn.Module):
    """
    Exact, differentiable 2D Discrete Cosine Transform (DCT-II) and Inverse (IDCT-III)
    using orthonormal transformation matrices.
    """
    def __init__(self, height=11, width=11):
        super(DCT2D, self).__init__()
        self.height = height
        self.width = width
        
        # Build 1D DCT matrices
        mat_h = self._build_dct_matrix(height)
        mat_w = self._build_dct_matrix(width)
        
        # Register as buffers (automatically moved to device with module)
        self.register_buffer("mat_h", mat_h)
        self.register_buffer("mat_w", mat_w)

    def _build_dct_matrix(self, N):
        matrix = np.zeros((N, N), dtype=np.float32)
        for k in range(N):
            for n in range(N):
                if k == 0:
                    matrix[k, n] = np.sqrt(1.0 / N)
                else:
                    matrix[k, n] = np.sqrt(2.0 / N) * np.cos(np.pi * (2 * n + 1) * k / (2.0 * N))
        return torch.from_numpy(matrix)

    def forward(self, x):
        """
        Forward 2D DCT: Y = M_h @ X @ M_w^T
        Input: [B, C, H, W]
        Output: [B, C, H, W]
        """
        B, C, H, W = x.shape
        # Dynamic check if spatial size differs from initialized size
        if H != self.height or W != self.width:
            mat_h = self._build_dct_matrix(H).to(x.device)
            mat_w = self._build_dct_matrix(W).to(x.device)
        else:
            mat_h = self.mat_h
            mat_w = self.mat_w
            
        # x: [B, C, H, W]
        # Y = mat_h @ x @ mat_w.T
        out = torch.matmul(mat_h, x)
        out = torch.matmul(out, mat_w.transpose(0, 1))
        return out

    def inverse(self, y):
        """
        Inverse 2D DCT (IDCT): X = M_h^T @ Y @ M_w
        Input: [B, C, H, W]
        Output: [B, C, H, W]
        """
        B, C, H, W = y.shape
        if H != self.height or W != self.width:
            mat_h = self._build_dct_matrix(H).to(y.device)
            mat_w = self._build_dct_matrix(W).to(y.device)
        else:
            mat_h = self.mat_h
            mat_w = self.mat_w
            
        # X = mat_h.T @ y @ mat_w
        out = torch.matmul(mat_h.transpose(0, 1), y)
        out = torch.matmul(out, mat_w)
        return out
