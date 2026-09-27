# B1–B4 特征提取实验

## VS Code 一键运行

1. 在 VS Code 打开项目，选择已有的 PyTorch 训练环境解释器。
2. 打开 `scripts/train_transmixer_ablation.py`，点击右上角 **Run Python File in Terminal**。
3. 脚本会顺序运行 B1 → B2 → B3 → B4，每个模型训练结束后自动加载最佳权重测试。无需分别启动测试脚本。

也可以在项目根目录终端运行：

```powershell
python scripts/train_transmixer_ablation.py
```

默认参数来自 `configs/feature_ablation.json`：Houston2013（编号 0），种子 6，100 epochs，学习率 0.0001，batch size 128，PCA 30，窗口 11，cuda:0。B0 不会重新运行。请先确认这些参数与已有 B0 一致。

数据沿用原框架的读取位置：项目 `data/` 或上一级 `data/`。Houston2013 需要 `Houston2013/houston_hsi.mat`、`houston_lidar.mat`、`houston_gt.mat`、`houston_index.mat`。也支持原框架的数据集内嵌 `data/` 结构。启动前会检查文件与 GPU；CUDA 不可用时不会悄悄改用 CPU。

## 四个独立改动

全部保留原始 1:1 分组、64 隐藏通道、两个 block、原始 stem、融合和分类头。每个 block 独立拥有自己的缩放参数。u 表示 LayerNorm 后的 SSM 输入，z 表示 SSM 输出，后半通道记为 u_l/z_l。

| 名称 | 相对 B0 的唯一结构/初始化改动 |
|---|---|
| TransMixer_B1_LocalResidual | local = Conv(z_l) + alpha × z_l；alpha 初始化 0 |
| TransMixer_B2_PreSSMResidual | local = Conv(z_l) + alpha × u_l；alpha 初始化 0 |
| TransMixer_B3_DeltaInit | 只改变 dt_proj.bias：softplus(bias) 的目标在 0.001～0.1 对数均匀初始化；保留原来的权重与 A |
| TransMixer_B4_MultiScaleDW | 在原 BN 前将 DWConv3×3 换成 DWConv3×3 + beta × DWConv5×5；beta 初始化 0 |

B3 的范围是偏置的 softplus 初始化目标，不是训练过程中实际 delta 的硬限制，也不是完整替换为官方 Mamba 实现。参考：https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py

网络先完整构造 B0，再复制共享参数并增加单项改动。相同种子下共享参数一致，新增模块构造也不推进外部 PyTorch RNG。B1、B2、B4 初始输出与 B0 一致；这不意味着完整训练过程或最终精度一定一致。B4 在 beta=0 时新卷积第一步梯度为 0，但 beta 可以学习，打开后新卷积即可更新。

## 修改实验组合

通常只需编辑 `configs/feature_ablation.json`：`models`、`datasets`、`seeds` 以及 `training`。数据集编号沿用原框架：0 Houston2013、1 Houston2018、2 Trento、3 Berlin、4 Augsburg、5 YellowRiverEstuary、6 LN01、7 LN02。

```powershell
# 仅预览任务清单，不训练、不创建输出文件，也不检查数据或 GPU
python scripts/train_transmixer_ablation.py --dry-run

# 只运行 B2/B3
python scripts/train_transmixer_ablation.py --models B2 B3

# 补两个种子；自动运行 8 项实验
python scripts/train_transmixer_ablation.py --seeds 7 8

# 指定数据集
python scripts/train_transmixer_ablation.py --datasets 0 3

# 真实数据的 1 epoch 冒烟验证（成绩不能与 100 epoch B0 比较）
python scripts/train_transmixer_ablation.py --epochs 1
```

如果确实需要 CPU，显式使用 `--device cpu`。每次启动都会生成一个新的时间标签，不会自动恢复上次训练。失败项会记录错误，继续后续模型；Ctrl+C 停止当前进程和剩余队列。重跑失败模型使用 `--models`，不必重跑成功项。

## 输出

- 权重：`model/TransMixer_B1_LocalResidual/` 等各模型目录。
- 原框架训练日志：`log/TransMixer_B1_LocalResidual/` 等。
- 分类报告：`report/TransMixer_B1_LocalResidual/` 等，含 OA、AA、Kappa 和逐类指标。
- 批次目录：`report/feature_ablation/<时间标签>/`，含每项完整配置、控制台日志、运行环境及源码哈希 `manifest.json`、逐项更新的 `summary.csv` 和 `summary.json`。

文件名包含批次标签与种子。一个失败实验不会被记为成功；重复标签/已有输出会被拒绝覆盖。

本轮沿用现有 trainer 的 **测试集精度选最佳权重** 协议，以便对照按相同协议得到的旧 B0。这批结果属于原协议下的探索性比较；若改为独立验证集选模型，B0 与变体需共同切换协议，不能混用两种结果。原框架的优化器、数据预处理、评估频率等没有修改。

## 本地验证

```powershell
python -m unittest discover -s src/models/transmixer_ablation -p 'test_feature_ablation.py' -v
```

测试覆盖共享初始化、零缩放时与 B0 的输出一致性、B1/B2 旁路差别、B3 步长偏置范围、B4 新分支梯度打开、两种模态输入的前向/反向、整模型权重保存加载，以及批量脚本的配置与失败记录。合成输入测试不能证明真实数据精度提升。
