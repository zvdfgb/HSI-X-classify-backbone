import torch
from model import DFFNet

# 1. 制造假数据（假设 BatchSize=4，高光谱 30 通道，LiDAR 1 通道，切片 11x11）
fake_hsi = torch.randn(4, 30, 11, 11)
fake_aux = torch.randn(4, 1, 11, 11)

# 2. 实例化模型（假设 15 个类别）
net = DFFNet(hsi_channels=30, aux_channels=1, num_classes=15, window_size=11)

# 3. 前向传播测试（看能不能算出得分）
out = net(fake_hsi, fake_aux)
print("输出形状:", out.shape)  # 正常应该打印 torch.Size([4, 15])

# 4. 反向传播测试（看梯度能不能正常回传，会不会报错或出 NaN）
loss = out.sum()
loss.backward()
print("反向传播测试通过！梯度计算正常！")