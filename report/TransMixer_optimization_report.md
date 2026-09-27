# TransMixer 系列模型整理与实验分析报告

报告日期：2026-09-27
项目分支：`feat/transmixer`
实验数据集：Houston2013
当前可复核实验：seed=6，PCA=30，window=11，batch size=128，learning rate=0.0001，epochs=100，checkpoint 仍按实验室原 trainer 的测试集精度选择。

## 1. 整理结论

当前分支已经统一为实验室 backbone 的工作方式：

```text
scripts/train.py / scripts/train_all.py
        ↓
src/runner.py
        ↓
src/config.py
        ↓
src/trainer.py / src/evaluator.py
        ↓
src/data.py
        ↓
src/model_registry.py
        ↓
src/models/<model>/adapter.py
```

B1–B4 只在模型适配层和模型结构中引入单项特征提取改动，训练、测试、日志、报告和分类图全部继续使用原 backbone 的统一流程。根目录的批量脚本已经归入 `scripts/`，模型测试放回对应的 `src/models/transmixer_ablation/`，说明与研究报告归入 `report/`。

## 2. TransMixer 规范命名

| 原名称或实验代号 | 规范名称 | 代码模式 | 当前状态 |
|---|---|---|---|
| `TransMixerNet` | `TransMixer_B0_Decoupled` | 固定 1:1 解耦 | 作为已有基线参照 |
| `TransMixer_B1` | `TransMixer_B1_LocalResidual` | B1 | 已训练并测试 |
| `TransMixer_B2` | `TransMixer_B2_PreSSMResidual` | B2 | 已训练并测试 |
| `TransMixer_B3` | `TransMixer_B3_DeltaInit` | B3 | 已训练并测试 |
| `TransMixer_B4` | `TransMixer_B4_MultiScaleDW` | B4 | 已训练并测试 |
| `TransMixer_Dyn` | `TransMixer_DynamicSplit` | Delta 排序动态分流 | 代码已接入，当前未发现本轮指标 |
| `TransMixer_Soft` | `TransMixer_SoftChannelGate` | Delta 驱动软门控 | 代码已接入，当前未发现本轮指标 |
| `TransMixer_Gate` | `TransMixer_CrossStreamGate` | 双流交叉门控 | 代码已接入，当前未发现本轮指标 |
| `TransMixer_DynGate` | `TransMixer_HybridGate` | 软门控加交叉门控 | 代码已接入，当前未发现本轮指标 |
| `TransMixer_Conv` | `TransMixer_LocalConv` | 纯局部深度卷积对照 | 代码已接入，当前未发现本轮指标 |
| `TransMixer_Attn` | `TransMixer_GlobalAttention` | 纯全局自注意力对照 | 代码已接入，当前未发现本轮指标 |
| `TransMixer_SSM` | `TransMixer_PureSSM` | 纯状态空间扫描对照 | 代码已接入，当前未发现本轮指标 |
| `TransMixer_DEG` | `TransMixer_DEG` | TransMixer 加方向引导边缘门控卷积 | 代码已接入，当前未发现本轮指标 |
| `DEGConvNet` | `TransMixer_DEGOnly` | 纯 DEGConv 主干 | 代码已接入，当前未发现本轮指标 |

旧名称没有被硬删除。`src/model_registry.py` 通过别名兼容旧配置，新实验只使用规范名称，避免批量脚本因为同一模型的多个别名而重复训练。

## 3. 当前可复核准确率

当前工作目录中实际保存并核对过的标准报告来自 B1–B4。B0 的 OA=88.9727% 是此前实验记录中的参照值；当前工作目录没有发现对应的 B0 checkpoint 和标准报告，因此这里不把它写成已经在本轮重新验证过的结果。

| 规范模型 | OA | AA | Kappa | 相对 B0 的 OA | 结果判断 |
|---|---:|---:|---:|---:|---|
| `TransMixer_B0_Decoupled` | 88.9727 | 未在本轮报告中保留 | 未在本轮报告中保留 | 0 | 已有基线记录，未在本轮重跑 |
| `TransMixer_B1_LocalResidual` | 88.9399 | 90.5873 | 87.9889 | -0.0328 | 与 B0 基本持平，未形成提升 |
| `TransMixer_B2_PreSSMResidual` | 88.9317 | 90.6062 | 87.9799 | -0.0410 | 与 B0 基本持平，未形成提升 |
| `TransMixer_B3_DeltaInit` | 88.4644 | 90.2942 | 87.4734 | -0.5083 | 当前配置下下降最明显 |
| `TransMixer_B4_MultiScaleDW` | 88.9891 | 90.5282 | 88.0428 | +0.0164 | 当前四个变体中最高，但提升极小 |

本轮只有一个随机种子，因此 B4 只能称为当前单次实验的最高结果，不能据此确认稳定增益。B1、B2 与 B0 的差距都低于 0.05 个百分点，优先应被视为同一性能水平。B3 的下降幅度相对明显，值得优先检查初始化策略与训练稳定性。

## 4. 各模型的创新点、缺点与分析

### `TransMixer_B0_Decoupled`

创新点是把 SSM 输出的通道固定按 1:1 分成全局流和局部流：前半部分使用多头自注意力，后半部分使用深度可分离卷积，再通过点卷积融合。它提供了稳定的全局语义与局部纹理分工，是当前所有特征提取优化的共同参照。

主要缺点是通道分工固定，样本之间不能根据内容改变全局流和局部流的比例，也不能根据不同通道的动态状态决定其更适合哪条分支。因此，B0 的结果稳定，但结构表达能力仍有进一步优化空间。

### `TransMixer_B1_LocalResidual`

B1 在局部卷积结果后增加原始局部 SSM 特征的可学习残差旁路：`Conv(z_l) + alpha · z_l`，其中 `alpha` 从 0 开始。它保留局部卷积提取的空间结构，同时避免卷积分支完全覆盖 SSM 的局部信息。

缺点是旁路特征与卷积输出存在较强重叠，新增的单个缩放参数提供的自由度很小。当前 OA 比 B0 低 0.0328 个百分点，说明该旁路没有带来可观的有效信息，可能只是增加了特征冗余。

### `TransMixer_B2_PreSSMResidual`

B2 将归一化后、进入 SSM 前的局部通道直接作为可学习旁路：`Conv(z_l) + alpha · u_l`。相比 B1，它保留了更早期的局部信息，理论上可以减少 SSM 扫描对短程细节的平滑影响。

缺点是它绕过了 SSM 的状态建模和动态步长调制，更容易把噪声或未充分建模的局部响应送到融合层。当前 OA 比 B0 低 0.0410 个百分点，表现与 B1 接近，暂时没有证据说明提前旁路优于 SSM 后旁路。

### `TransMixer_B3_DeltaInit`

B3 只改变 `dt_proj.bias` 的初始化，使 `softplus(bias)` 对应的目标步长在 0.001 到 0.1 之间对数均匀分布，同时保留其它 B0 权重。该设计试图让不同通道在训练初期具备更宽的状态记忆时间尺度。

缺点是它只改变初始化，没有同步调整学习率、状态矩阵或步长约束。步长初始化与共享权重之间可能不匹配，导致早期状态更新尺度改变，训练过程更难稳定。当前 OA 比 B0 低 0.5083 个百分点，是本轮最需要谨慎对待的变体；在没有多 seed 证据前，不建议把它作为有效改进方向。

### `TransMixer_B4_MultiScaleDW`

B4 在原有 3×3 深度卷积路径上增加 5×5 深度卷积路径：`DWConv3×3 + beta · DWConv5×5`，并将 `beta` 初始化为 0。这样可以在不破坏 B0 初始行为的情况下，为局部空间结构增加更大的感受野。

缺点是参数量和计算量增加，且单个 `beta` 对所有通道共享，表达能力仍然有限。当前 OA 比 B0 高 0.0164 个百分点，提升小到可能处于随机波动范围；不过它是 B1–B4 中最值得继续做多 seed 验证的方向，因为改动位置明确、初始行为连续、可解释性较好。

### `TransMixer_DynamicSplit`

该模型根据 SSM 输出的平均 `Delta_t` 对通道排序，把较小步长通道分配给全局注意力，把较大步长通道分配给局部卷积，再用 gather/scatter 恢复原通道位置。它把固定通道解耦改成了样本自适应分流。

主要风险是排序在 `no_grad` 环境中进行，通道分组本身不可微；gather/scatter 也带来额外的显存访问和实现复杂度。它适合做结构验证，但需要记录运行时间、显存和跨 seed 稳定性，不能只看单次 OA。

### `TransMixer_SoftChannelGate`

该模型由 `Delta_t` 生成连续通道门控，对全局流和局部流进行软缩放，避免动态排序的离散切换。它理论上比 DynamicSplit 更容易优化，也更适合表达通道重要性差异。

主要风险是当前实现对两个流使用互补缩放，但没有显式的稀疏约束或熵约束，门控可能趋向接近常数，最后退化成 B0 的近似形式；也可能出现门控饱和，导致某一流长期占优。需要额外保存门控统计量才能判断它是否真正学习到了分工。

### `TransMixer_CrossStreamGate`

该模型用全局流生成局部门控，用局部流生成全局门控，使两条分支能够互相校准，目标是缓解固定解耦后两条流之间的信息隔离。

主要风险是互相放大的门控会改变两个分支的数值尺度，并可能造成错误信息传播。它还增加了两组线性门控层，若数据量有限，容易增加冗余参数。评估时应同时检查每条分支的激活分布和门控均值。

### `TransMixer_HybridGate`

该模型同时使用 Delta 驱动软门控和双流交叉门控，属于方向一与方向二的复合优化，理论上覆盖通道自适应分流和跨流互补校准两个问题。

主要缺点是改动叠加后难以判断收益来源，参数、梯度和尺度耦合也更复杂。如果单独的 SoftChannelGate 或 CrossStreamGate 没有稳定收益，直接使用 HybridGate 很可能只是叠加不稳定因素。因此它应在两个单项变体完成多 seed 验证后再作为最终模型。

### `TransMixer_LocalConv`、`TransMixer_GlobalAttention`、`TransMixer_PureSSM`

这三个模型是结构对照组，分别保留局部卷积、全局注意力或纯 SSM 扫描，用来回答性能是否来自某一单独机制。

- `TransMixer_LocalConv` 的优势是局部归纳偏置明确、计算路径简单；缺点是长距离依赖不足。
- `TransMixer_GlobalAttention` 的优势是全局交互直接；缺点是空间 token 数增大时计算和显存压力更高，并且缺少局部纹理先验。
- `TransMixer_PureSSM` 的优势是线性序列扫描和较长范围状态记忆；缺点是缺少显式的局部几何细化与全局注意力校正。

当前目录没有发现这三个对照模型对应的本轮 checkpoint、日志或标准报告，因此不填写虚构的准确率。

### `TransMixer_DEG` 与 `TransMixer_DEGOnly`

`TransMixer_DEG` 在 TransMixer 的局部流中加入方向引导边缘门控卷积，使用 Sobel 梯度先验、水平和垂直条形卷积以及动态门控，适合道路、铁路、建筑边缘等各向异性结构。`TransMixer_DEGOnly` 则作为纯 DEGConv 主干对照。

它们的主要风险是方向先验对数据集和场景分布敏感，条形卷积与边缘门控也会增加实现和调参成本。当前没有本轮指标，后续应和 B0、B4 使用完全相同的训练协议单独建立实验批次。

## 5. 输出和证据位置

当前 B1–B4 的标准 backbone 输出已经按规范名称整理：

```text
model/TransMixer_B1_LocalResidual/
model/TransMixer_B2_PreSSMResidual/
model/TransMixer_B3_DeltaInit/
model/TransMixer_B4_MultiScaleDW/

log/TransMixer_B1_LocalResidual/
log/TransMixer_B2_PreSSMResidual/
log/TransMixer_B3_DeltaInit/
log/TransMixer_B4_MultiScaleDW/

report/TransMixer_B1_LocalResidual/
report/TransMixer_B2_PreSSMResidual/
report/TransMixer_B3_DeltaInit/
report/TransMixer_B4_MultiScaleDW/

pic/TransMixer_B1_LocalResidual/
pic/TransMixer_B2_PreSSMResidual/
pic/TransMixer_B3_DeltaInit/
pic/TransMixer_B4_MultiScaleDW/
```

`report/feature_ablation/b1b4_20260927_095247_875122/` 保留了原始批次的 JSON 配置、控制台日志、运行环境、源码哈希和汇总文件，作为实验溯源记录。目录名和原始控制台文本保留旧代号是有意的，便于和训练当时的日志逐行对应；规范名称与旧代号的映射见本报告第 2 节。

## 6. 结论与下一步

当前证据支持的结论是：固定 1:1 B0 仍然是稳定参照；B1、B2 没有显示有效提升；B3 在当前初始化策略下有明显退化；B4 是最值得继续验证的候选，但单次提升不足以证明有效。

建议下一批实验保持所有 backbone 参数、数据预处理和 checkpoint 选择协议不变，只增加 seeds `[6, 7, 8]`，优先比较 `TransMixer_B0_Decoupled`、`TransMixer_B3_DeltaInit` 和 `TransMixer_B4_MultiScaleDW`。对于 DynamicSplit、SoftChannelGate、CrossStreamGate 和 HybridGate，除了 OA、AA、Kappa，还应记录单 batch 推理时间、峰值显存、门控均值以及不同类别的召回率。
