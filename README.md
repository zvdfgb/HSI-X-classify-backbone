# HSI-X-Classify-Backbone

当前研究方向：固定1:1局部／全局通道分工＋SR邻域加权汇总，以及原生SR引导的轻量邻域融合。已批准的`K00–K06`与`R00–R05`三种子实验均已完成，当前三种子平均OA最高的候选为`R01`。后续训练严格使用实验室原入口和原规则，新增实验须先审核；详见[最新研究记录（更新至2026-11-09）](docs/RESEARCH_PROGRESS_20261105.md)、[项目目录与训练导航](docs/PROJECT_GUIDE.md)和[代码复用规则](AGENTS.md)。

2026-11-08批次的`A0`与`C02`原生SR引导邻域融合双基座对照（`R00–R05`，`seed6/7/8`共18项新训、复用`A0/K01–K03/C02`共15项）已于**2026-11-09 00:15（北京时间）**完成：见[完整证据归档](docs/results/fusion_sr_a0_c02_seed678_20261108/README.md)、[实验报告](docs/results/fusion_sr_a0_c02_seed678_20261108/实验报告.md)与[11月4—9日研究记录](docs/RESEARCH_PROGRESS_20261105.md)。

- **结构与参数量：**原`A0`为`80,887`参数，融合层是`128 -> 64`的`1×1`卷积。`R00–R02`在该融合层之前新增两路：`128 -> 64`的`1×1`中心路，以及先经`128 -> 64`的`1×1`压缩、再经`64 -> 64`的密集`3×3`卷积的邻域路；两路拼接后仍送入原融合层。`R01`仅在邻域路加入正确`SR2`权重调制，三组参数量均为`134,519`。`R01`较`C02`（`680,311`参数）少`80.2%`参数，但较原`A0`增加`66.3%`，不应描述为相对`A0`压缩。
- **整体结果：**`R01`取得当前已归档原流程研究结果中最高的三种子平均OA：**`91.2929 ± 1.6286%`**；AA为`92.6780 ± 1.2499%`，Kappa为`90.5457 ± 1.7684%`。相对同结构均匀对照`R00`（`90.4949%`）为`+0.7980`个百分点（OA `2/3`正向），相对融合先验固定错位对照`R02`（`90.4348%`）为`+0.8581`个百分点（OA `2/3`、AA `3/3`正向），相对原`A0`（`89.3990%`）为`+1.8939`个百分点（`2/3`正向，`seed6`为`-2.2793`），相对`K02`（`89.9101%`）为`+1.3829`个百分点（`3/3`正向），相对`C02`（`90.8256%`）为`+0.4673`个百分点（`2/3`正向）。`R00 - A0`已为`+1.0959`个百分点，因此不能把`R01 - A0`的全部提升归因于SR引导。
- **逐类结果：**按原生分类报告打印值汇总，`Road` F1为`89.07%`，较`R00`/`A0`分别约`+8.04`/`+7.95`个百分点；`Highway`召回率为`90.31%`，较`R00`/`C02`分别约`+12.39`/`+12.55`个百分点，以上四项均为`3/3`正向。差值先用未舍入的三种子均值计算，再保留两位小数。
- **C02基座对照：**零新增参数的`R04`（`90.5414%`）低于同路径均匀对照`R03`（`90.6480%`）、错位对照`R05`（`90.6234%`）及原`C02`（`90.8256%`）；相对`R03`，`Commercial`召回率下降`4.84`个百分点，三个种子均下降，触发该基座的预设证伪判据。结果支持SR引导的收益依赖融合结构，尚不能据此认定“过度中心同质性约束”为已证实的因果机制。

2026-11-07批次的核共享紧凑多尺度与SR引导跨模态邻域融合（`SR-NCF`，`K00–K06`，`seed6/7/8`共21项新训、复用`A0/M1/N00/C02`共12项）已于2026-11-08完成：见[完整证据归档](docs/results/fusion_sr_ncf_seed678_20261107/README.md)、[实验报告](docs/results/fusion_sr_ncf_seed678_20261107/实验报告.md)与[11月4—9日研究记录](docs/RESEARCH_PROGRESS_20261105.md)。

- 核共享多尺度＋读出`SR2`的`K01`为`108,111`参数，平均OA为`89.0629 ± 2.1350%`；相对`A0`为`-0.3361`个百分点（`2/3`正向），相对近等参数预算的独立三分支`K00`（`108,615`参数，`88.6584 ± 2.0364%`）为`+0.4045`个百分点（仅`1/3`正向）。
- 无末端`SR2`、使用`GAP`读出时，融合阶段引入正确`SR2`（`K04 - K06`）带来`+0.8718`个百分点（`3/3`正向）。有末端`SR2`时，叠加融合`SR2`的`K02`平均OA为`89.9101 ± 1.2790%`，相对均匀权重`K01`为`+0.8472`个百分点（仅`1/3`正向），相对错位对照`K03`（`89.3936%`）为`+0.5165`个百分点（`2/3`正向）。
- 四格交互描述项`(K02 - K01) - (K04 - K06)`的平均OA差为`-0.0246`个百分点，不支持“强负交互”结论。当前证据表明：融合SR在无末端SR读出时收益较一致，在已有末端SR读出时收益受种子影响；不将该描述项解释为统计显著或因果证明。

以上K/R结果均限于Houston2013、`seed6/7/8`和原测试OA选模规则；`±`为三种子的样本标准差，不是置信区间，不构成独立测试、统计显著性或SOTA证明。

2026-11-07轻量化BMAF分通道密集多尺度与边界感知融合完成：针对`HAMF`三项失效根因修正的`BMAF`（`N00–N05`，`seed6/7/8`共18项新训、复用`A0/C01/C02`共9项）已完成并复核通过，见[完整证据归档](docs/results/fusion_bmaf_seed678_20261106/README.md)与[11月4—7日研究记录](docs/RESEARCH_PROGRESS_20261105.md)。核心轻量化分通道密集多尺度基座`N02`（`182,631`参数，较`C02`减重`73.2%`、每patch估计MACs降低`69.1%`）三种子平均OA达`90.0167 ± 0.6937%`，突破`90%`门槛：较`A0`（`89.3990 ± 2.6142%`）提升`+0.6176`个百分点且标准差收窄`73.5%`，较重参数单尺度`C01`（`89.8937%`，`253,559`参数）提升`+0.1230`个百分点，较同结构单尺度控制组`N01`提升`+0.6477`个百分点（AA提升`+0.6916`，`3/3`同向），较无跨模态互导的`N00`提升`+0.1804`个百分点。边界调制主候选`N04`（`89.7270 ± 1.1684%`）虽优于`A0`（`+0.3279`）、无先验调制`N03`（`+0.1284`）与空间错位对照`N05`（`+0.1558`，`2/3`正向），并改善了`Road`（较`N02` `+1.26`个百分点）与`Highway`（`+1.16`），但整体仍低于无门控固定拼接`N02`（`-0.2897`个百分点），验证了在小样本纯CE训练下无方向约束像素级动态门控易产生上下文捷径过拟合。

2026-11-06轻量化HAMF验证与阶段总结：针对`C02`参数重与边界模糊设计的轻量化`HAMF`（`M00–M05`，`seed6/7/8`共18项新训）已完成，见[完整证据归档](docs/results/fusion_hamf_seed678_20261105/README.md)与[11月4—7日研究记录](docs/RESEARCH_PROGRESS_20261105.md)。主候选`M04`平均OA为88.8634%（`M04-A0`为`-0.5357`个百分点），未能复现`C02`（90.8256%）的增益；步级诊断定位出三大根因：`lr=1e-4`下双重`0.1`残差标量受2200步上限钳制（终值仅0.17–0.24，使83%信号走恒等捷径）、`groups=128`逐通道深度卷积割裂了HSI与LiDAR的跨模态空间邻域耦合、以及全窗归一化`W_sr`在细窄`Road`中心的反相峰值效应。

2026-11-05融合迁移总结：第三批三篇新论文融合子机制迁移（`CF-IPT`、`ISFuse`、`IID`）39/39项（复用A0三项、新训36项）已完成独立复核，见[11月4—6日研究记录](docs/RESEARCH_PROGRESS_20261105.md)和[完整证据归档](docs/results/fusion_transfer_v3_seed678_20261104/README.md)。三个预设完整候选（`C04` 89.8882%、`S04` 88.5710%、`I04` 88.7541%）均未同时优于`A0`（89.3990%）与各自控制组；次要控制组`C02`（无通道门控的多尺度交互）平均OA达90.8256%，`C02-C01`（+0.9319）与`C04-C03`（+0.7352）在三个种子上均同向为正。

2026-11-04机制分析：15个历史模型的只读诊断及LL／GG各seed6/7/8共6次原流程新训均已完成，见[详细分析](docs/experiments/20261104_局部全局与SR机制分析.md)和[完整证据归档](docs/results/sr_lg_mechanisms_20261104/README.md)。混合LG、两路局部LL、两路全局GG的平均OA分别为89.3990%、87.5243%、89.4100%；LG相对LL为3/3正向，相对GG仅1/3。SR独立收益较稳定；固定1∶1混合的整体优势未被证明，但道路类相对LL／GG均有3/3正向线索，见[类别补充分析](docs/results/sr_lg_mechanisms_20261104/算子对照类别补充分析.md)。训练与本批临时防休眠均已结束，原测试OA选模规则不变。

上一批进度（2026-11-03）：这两天已完成三篇论文融合机制迁移36次及SR先验协同6次新训；连同复用的3个A基线，共45项不重复结果，全部使用seed6/7/8。详见[两日研究总结](docs/RESEARCH_PROGRESS_20261103.md)、[45项逐种子CSV](docs/results/fusion_20261102_03_results.csv)和[结果目录](docs/results/README.md)。该批训练与守护均已结束。

| 方向／完整候选 | 三种子平均OA % | 相对A00平均OA差（百分点） | 正向种子数 |
|---|---:|---:|---:|
| A00：固定1:1局部／全局＋SR2 | 89.3990 | — | — |
| F04：三频带融合＋高频补偿 | 89.1503 | −0.2487 | 0/3（1项持平） |
| U03：概率表示＋空间置信融合 | 87.1007 | −2.2984 | 0/3 |
| D04：共享／特有分解＋任务约束 | 88.6775 | −0.7215 | 0/3 |
| P01：正确SR先验进入全局注意力 | 89.4537 | +0.0547 | 3/3 |
| P02：固定错位先验对照 | 89.4018 | +0.0027 | 2/3 |

F/U/D完整候选未提升原A；P01相对P02平均OA +0.0519个百分点，仅是很小的正向线索。全部均保留原测试OA选模，不构成独立测试或统计显著性证明。原始数据、完整日志与checkpoint留在本机；Git收录代码、配置及带哈希的轻量报告证据。

B1三种子也已完成，见[最终18项快照](docs/results/fusion_B1_seed78_20261101_final_20261103/README.md)：在纯SSM上平均OA +0.9374个百分点，在A上−0.1202。原[11月1日进度](docs/RESEARCH_PROGRESS_20261101.md)及10/18快照按历史状态保留。

此前三篇论文迁移实验27项的历史结果保留、不重跑，见 [论文迁移结果](docs/results/paper_20261028/README.md)。不同训练协议的成绩分别报告。

助手重复编写的旧训练流程、批量脚本和配置已 [归档](docs/archive/assistant_experiments_20261029/README.md)，不再作为日常入口。此前不同协议的实验见 [周末实验结果](docs/results/weekend_20261024_25/README.md)。

用于高光谱与辅助模态（LiDAR / SAR / MSI）融合分类的统一实验骨架。项目提供了统一的实验配置、数据加载、模型注册、训练、评估与可视化流程，便于在多个公开数据集上复现实验、横向对比不同模型，并继续扩展新模型。

如果本项目对你的研究有所帮助，欢迎引用我们的最新工作：

```bibtex
@ARTICLE{rscnet26tgrs,
  author={Gong, Chuanzheng and Gao, Feng and Lin, Junyan and Dong, Junyu and Du, Qian},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  title={Representative Spectral Correlation Network for Multisource Remote Sensing Image Classification},
  year={2026},
  volume={64},
  pages={1-14}
}
```

同时，我们的最新工作也已经开源，欢迎关注 [oucailab/RSCNet](https://github.com/oucailab/RSCNet)。

---

## 1. 项目概览

本项目面向 HSI-X 多模态遥感分类任务，目标是把“数据集切换、模型切换、训练测试、结果落盘”统一到一套最小可维护框架中。

当前代码库已经将原先分散的参数修改逻辑收敛到统一配置系统中，核心特点如下：

- 统一实验配置：通过 `src/config.py` 中的 `ExperimentConfig` 管理数据集、模型、训练参数和输出路径。
- 统一模型注册：通过 `src/model_registry.py` 管理可用模型与适配器模块。
- 统一训练/测试入口：通过 `src/runner.py` 组织训练、测试、可视化、t-SNE 等流程。
- 统一数据接口：通过 `src/data.py` 适配多个多模态遥感数据集。
- 统一结果输出：模型权重、日志、报告、可视化图像按照模型名和实验参数自动归档。

项目适合以下场景：

- 复现实验室已有模型结果
- 在同一批数据集上对比多个 backbone
- 在已有框架中快速接入新模型
- 批量训练多模型 / 多数据集实验

---

## 2. 项目架构

### 2.1 目录结构

```text
HSI-X-classify-backbone/
├── configs/                  # 训练等脚本默认读取的配置文件目录
├── data/                     # 本机数据，不纳入Git
├── environment.yml           # 通用环境模板；批次实际环境见RUNTIME.json
├── requirements.txt          # 精简 pip 依赖列表
├── Makefile                  # 常用命令封装
├── log/                      # 训练日志输出
├── model/                    # 模型权重输出
├── pic/                      # 分类可视化图输出
├── report/                   # 分类报告输出
├── docs/                     # 研究总结、审核矩阵与结果证据导航
├── scripts/                  # 原训练入口、已批准批次控制器、只读核验
└── src/
    ├── config.py             # 统一配置系统
    ├── data.py               # 数据读取、PCA、归一化、DataLoader
    ├── runner.py             # 实验调度入口
    ├── trainer.py            # 训练流程
    ├── evaluator.py          # 测试流程
    ├── metrics.py            # 分类指标与报告输出
    ├── analysis.py           # 分类图可视化
    ├── tsne.py               # t-SNE 分析
    ├── show_erf.py           # ERF 相关入口
    ├── visualize_erf.py      # ERF 可视化
    ├── model_registry.py     # 模型注册表
    └── models/
        ├── fusatnet/
        ├── exvit/
        ├── hybridsn/
        ├── micf_net/
        ├── cnn/
        ├── fdnet/
        ├── s2enet/
        ├── asyffnet/
        ├── hctnet/
        ├── macn/
        ├── rscnet/
        ├── transmixer_net/
        ├── transmixer_sr/
        ├── transmixer_fusion_papers/
        ├── transmixer_fusion_v2/
        ├── transmixer_sr_prior/
        ├── transmixer_ablation/
        └── transmixer_deg/
```

### 2.2 核心执行链路

一次完整实验的调用链路如下：

```text
scripts/train.py / scripts/test.py / scripts/train_all.py
        ↓
src.runner.ExperimentRunner
        ↓
src.config.ExperimentConfig / ConfigManager
        ↓
src.trainer.myTrain / src.evaluator.myTest
        ↓
src.data.getMyData
        ↓
src.model_registry.get_model_adapter
        ↓
src.models.<model>.adapter
        ↓
具体模型实现 + 指标统计 + 可视化输出
```

### 2.3 模块职责说明

#### 统一配置层
- [src/config.py](src/config.py)
  - 定义 `ExperimentConfig`
  - 维护数据集编号到名称的映射
  - 自动构建模型、日志、报告、图片输出路径
  - 提供 CLI 参数解析与配置文件保存/加载能力

#### 数据层
- [src/data.py](src/data.py)
  - 负责 `.mat` 数据读取
  - 对 HSI 做 PCA 降维
  - 对 HSI / 辅助模态做标准化
  - 生成训练、测试、可视化所需的 `DataLoader`
  - 内置 8 个数据集的数据文件约定与 key 约定

#### 模型适配层
- [src/model_registry.py](src/model_registry.py)
  - 将模型名映射到 `src.models.<name>.adapter`
- `src/models/*/adapter.py`
  - 负责屏蔽不同模型前向接口差异
  - 统一暴露 `build_model`、`forward_train`、`forward_eval`

#### 训练与评估层
- [src/trainer.py](src/trainer.py)
  - 原生训练、测试OA选模、整模型保存、日志记录
- [src/evaluator.py](src/evaluator.py)
  - 加载训练好的模型并生成报告
- [src/metrics.py](src/metrics.py)
  - 输出 OA / AA / Kappa / 分类报告 / 混淆矩阵
- [src/analysis.py](src/analysis.py)
  - 生成分类图像

---

## 3. 支持的数据集

数据集下载链接：[Google Drive](https://drive.google.com/drive/folders/1P9pr-6vbdnS4PPdo1fhg5jP9oZC4ct-v?usp=sharing)。

当前配置中内置了 8 个数据集编号，定义见 [src/config.py:20-29](src/config.py#L20-L29) 与 [src/data.py:270-320](src/data.py#L270-L320)。

| 编号 | 数据集 | 类别数 | HSI 通道数 | 辅助模态通道数 | 辅助模态类型 |
|---|---|---:|---:|---:|---|
| 0 | Houston2013 | 15 | 144 | 1 | LiDAR |
| 1 | Houston2018 | 20 | 50 | 1 | LiDAR |
| 2 | Trento | 6 | 63 | 1 | LiDAR |
| 3 | Berlin | 8 | 244 | 4 | SAR |
| 4 | Augsburg | 7 | 180 | 4 | SAR |
| 5 | YellowRiverEstuary | 18 | 285 | 3 | SAR |
| 6 | LN01 | 10 | 166 | 8 | MSI |
| 7 | LN02 | 9 | 144 | 8 | MSI |

### 3.1 数据文件放置方式

数据应放在项目根目录下的 `data/` 目录中，即 `HSI-X-classify-backbone/data/`。

推荐按下面方式组织数据：

```text
HSI-X-classify-backbone/
├── data/
│   ├── Houston2013/
│   │   ├── houston_hsi.mat
│   │   ├── houston_lidar.mat
│   │   ├── houston_gt.mat
│   │   └── houston_index.mat
│   ├── Houston2018/
│   ├── Trento/
│   ├── Berlin/
│   ├── Augsburg/
│   ├── YellowRiverEstuary/
│   ├── LN01/
│   └── LN02/
├── scripts/
├── src/
└── README.md
```

### 3.2 各数据集文件名约定

| 数据集 | 必需文件 |
|---|---|
| Houston2013 | `houston_hsi.mat`, `houston_lidar.mat`, `houston_gt.mat`, `houston_index.mat` |
| Houston2018 | `houston_hsi.mat`, `houston_lidar.mat`, `houston_gt.mat`, `houston_index.mat` |
| Trento | `trento_hsi.mat`, `trento_lidar.mat`, `trento_gt.mat`, `trento_index.mat` |
| Berlin | `berlin_hsi.mat`, `berlin_sar.mat`, `berlin_gt.mat`, `berlin_index.mat` |
| Augsburg | `augsburg_hsi.mat`, `augsburg_sar.mat`, `augsburg_gt.mat`, `augsburg_index.mat` |
| YellowRiverEstuary | `data_hsi.mat`, `data_sar.mat`, `data_gt.mat`, `data_index.mat` |
| LN01 | `LN01_HHSI.mat`, `LN01_MSI.mat`, `LN01_GT.mat`, `LN01_INDEX.mat` |
| LN02 | `LN02_HHSI.mat`, `LN02_MSI.mat`, `LN02_GT.mat`, `LN02_INDEX.mat` |

---

## 4. 支持的模型

当前注册模型以 [src/model_registry.py](src/model_registry.py) 为准。TransMixer 家族采用统一的 `TransMixer_` 前缀和功能后缀：

- `TransMixer_B0_Decoupled`：固定 1:1 光谱-空间解耦基线
- `TransMixer_B1_LocalResidual`：局部分支残差旁路
- `TransMixer_B2_PreSSMResidual`：SSM 前局部残差旁路
- `TransMixer_B3_DeltaInit`：Delta 步长初始化优化
- `TransMixer_B4_MultiScaleDW`：多尺度深度可分离卷积
- `TransMixer_DynamicSplit`、`TransMixer_SoftChannelGate`、`TransMixer_CrossStreamGate`、`TransMixer_HybridGate`
- `TransMixer_LocalConv`、`TransMixer_GlobalAttention`、`TransMixer_PureSSM`：结构对照组
- `TransMixer_DEG`、`TransMixer_DEGOnly`：方向引导边缘门控卷积及纯 DEGConv 对照

当前研究接入的模型还包括`TransMixer_SR1/SR2`、`TransMixer_FV2_F01`至`F04`、`U01`至`U04`、`D01`至`D04`（后三族均带`TransMixer_FV2_`前缀）、`TransMixer_SRP_P01/P02`、`TransMixer_FV3_C01`至`C04`／`S01`至`S04`／`I01`至`I04`、`TransMixer_HAMF_M00`至`M05`、`TransMixer_BMAF_N00`至`N05`、`TransMixer_SRNCF_K00`至`K06`以及`TransMixer_SRA0C02_R00`至`R05`。对应设计、已完成状态与使用边界见[实验导航](docs/experiments/README.md)。历史B1/B2/B3命名须按批次理解，不能把上述旧局部残差B1与后来的交错扫描B1混为同一个模型。

旧配置中的 `TransMixerNet`、`TransMixer_B1` 等名称仍可读取，会自动映射到上述规范名称；新实验只使用规范名称，避免 `train_all.py` 重复训练别名。

- `FusAtNet`
- `ExViT`
- `HybridSN`
- `MICF_Net`
- `CNN`
- `FDNet`
- `S2ENet`
- `AsyFFNet`
- `HCTNet`
- `MACN`
- `RSCNet`

每个模型目录通常包含：

```text
src/models/<model_name>/
├── adapter.py     # 与统一训练框架的接口桥接层
├── model.py       # 具体网络结构
└── __init__.py
```

适配器的作用可以参考：
- [src/models/fusatnet/adapter.py](src/models/fusatnet/adapter.py)
- [src/models/rscnet/adapter.py](src/models/rscnet/adapter.py)

其中：
- `build_model(...)` 负责根据数据集维度创建模型
- `forward_train(...)` 负责统一训练前向
- `forward_eval(...)` 负责统一推理前向

---

## 5. 环境安装

### 5.1 通用模板与批次实际环境

项目提供通用Conda模板[environment.yml](environment.yml)。2026-11-02—03研究批次实际使用Python3.13.5、PyTorch2.6.0+cu124、NumPy2.3.5和RTX4060 Laptop GPU，见[批次环境记录](docs/results/fusion_transfer_seed78_20261102/RUNTIME.json)。下面模板版本与实际批次不同，不能当作精确复现锁定文件。

关键信息如下：

- Python：`3.10.20`
- PyTorch：`2.4.1`
- torchvision：`0.19.1`
- torchaudio：`2.4.1`
- CUDA 运行时：`cu121`
- cuDNN：`9.1.0`

模板包含CUDA12.1依赖，例如：

- `torch==2.4.1+cu121`
- `torchvision==0.19.1+cu121`
- `torchaudio==2.4.1+cu121`
- `nvidia-cuda-runtime-cu12==12.1.105`
- `nvidia-cudnn-cu12==9.1.0.70`

安装前核对GPU驱动与Torch兼容性；精确追溯已有实验时，以对应批次RUNTIME.json及manifest为准。

### 5.2 使用 Conda 安装（推荐）

在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate HSI-X-Classify-Backbone
```

验证安装是否成功：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如果输出中包含 `2.4.1+cu121` 且 GPU 可用，则说明环境匹配成功。

### 5.3 使用 pip 安装（备选）

项目也提供了一个较精简的 [requirements.txt](requirements.txt)，但它**没有把 CUDA 版本锁死**，因此更适合你已经自行准备好 PyTorch GPU 环境后再补齐其余依赖。

推荐顺序：

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# 先安装与你机器匹配的 PyTorch GPU 版本
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# 再安装其余依赖
pip install -r requirements.txt
```

此处是通用安装示例；严格复查已有研究需核对对应批次环境，不能仅依赖模板。

### 5.4 常见环境说明

#### 4090 + CUDA 12.1
推荐直接使用仓库内 `environment.yml`。

#### 只有 CPU
CPU可用于接口检查；正式科研实验仍需批准配置。若要使用CPU，在独立配置副本中设置`cuda_device`为`cpu`，再用`python scripts/train.py --config <配置路径>`运行；该入口不支持逐项传入`--cuda`。

#### 已有不同 CUDA 版本
如果你本机 PyTorch 来自其他 CUDA 版本，请注意：
- `requirements.txt` 不会自动帮你切到 `cu121`
- 不同 CUDA 版本下，最好重新安装与本机驱动匹配的 Torch 三件套

---

## 6. 运行方式

### 6.1 Makefile 方式

项目根目录提供了 [Makefile](Makefile)，可直接查看帮助：

```bash
make help
```

常用命令：

```bash
make activate
make train
make test
make strain
make stest
make train-all
```

说明：
- `make train` 调用 [scripts/train.py](scripts/train.py)
- `make test` 调用 [scripts/test.py](scripts/test.py)
- `make train-all` 调用 [scripts/train_all.py](scripts/train_all.py)

如需传参：

```bash
make train ARGS="--help"
```

> 当前原生训练入口支持`--config`，默认读取`configs/train.json`。研究实验使用批准的独立配置文件，不修改入口或另写训练循环。

### 6.2 直接运行脚本

#### 训练

```bash
python scripts/train.py
```

当前 `scripts/train.py` 的行为是：

- 默认优先查找 `configs/train.json`
- 如果该配置文件存在，则直接加载它作为训练配置
- 如果该配置文件不存在，则回退到脚本内置的 quick config
- 不再支持通过命令行逐项覆盖训练参数

当前内置 quick config 默认值为：
- 数据集：`Augsburg`
- 模型：`RSCNet`
- 学习率：`1e-4`
- Epochs：`100`
- Batch size：`128`
- 默认开启测试与可视化

常见示例：

```bash
# 直接运行：优先读取 configs/train.json，不存在则走 quick config
python scripts/train.py

# 指定新审核的独立配置文件（替换为实际路径）
python scripts/train.py --config <已批准的新配置路径>
```

原`configs/train.json`作为默认标准保留；研究实验从它派生到`configs/runs/<新批次>/`，不原地覆盖默认或已封存配置。

配置文件里只需要保留你真正想控制的实验参数，例如数据集、模型、学习率、epoch、batch size、窗口大小、是否可视化等。

像下面这些字段**不建议写进配置文件**，因为框架会自动推导：

- `out_features`
- `data_channels`
- `lidar_or_sar_channels`
- `depth`（如果你接受默认结构）
- `model_save_path`
- `log_path`
- `report_path`
- `image_path`
- `image_path_web`

其中输出路径会根据当前机器上的项目根目录、模型名、数据集名和实验参数自动生成，因此不要硬编码绝对路径。

#### 测试

```bash
python scripts/test.py
```

当前默认配置见 [scripts/test.py:12-23](scripts/test.py#L12-L23)：
- 数据集：`Houston2013`
- 模型：`RSCNet`
- Epochs：`10`
- 仅测试，不训练

#### 批量训练

```bash
python scripts/train_all.py
```

可用参数：

```bash
python scripts/train_all.py -h
```

常见示例：

```bash
# 对全部模型、全部数据集训练
python scripts/train_all.py

# 只跑指定模型
python scripts/train_all.py --models RSCNet FusAtNet --epochs 100 --cuda cuda:0

# 只跑指定数据集（可用编号或名称）
python scripts/train_all.py --datasets 0 4 LN01

# 指定实验名前缀
python scripts/train_all.py --experiment-prefix paper_baseline
```

`scripts/train_all.py` 会自动检查数据是否存在，缺失时跳过，不会直接崩掉。

### 6.3 Python API 方式

项目支持程序化调用，可直接使用 [src/runner.py](src/runner.py) 中的接口。

#### 快速运行

```python
from src.runner import quick_run

quick_run(
    dataset_type=0,
    model_name="FusAtNet",
    lr=1e-4,
    epochs=10,
    channels=30,
    window_size=11,
    cuda_device="cuda:0",
    enable_visualization=True,
    enable_testing=True,
    experiment_name="quick_demo",
)
```

#### 基于配置对象运行

```python
from src.config import ExperimentConfig
from src.runner import ExperimentRunner

config = ExperimentConfig(
    experiment_name="custom_exp",
    dataset_type=4,
    model_name="RSCNet",
    learning_rate=1e-4,
    epochs=100,
    batch_size=128,
    channels=30,
    window_size=11,
    cuda_device="cuda:0",
    enable_training=True,
    enable_testing=True,
    enable_visualization=True,
    enable_tsne=False,
)

runner = ExperimentRunner(config)
runner.print_config()
runner.run_experiment()
```

### 6.4 CLI 配置方式

原生入口已经支持配置文件，无需新增CLI或训练入口：

```bash
python scripts/train.py --help
python scripts/train.py --config <新批准的逐组逐种子配置>
```

模型、seed、设备和训练参数通过JSON配置提供；当前研究配置在`configs/runs/`按批次保存。已有完成批次禁止覆盖重跑，下一轮仍先审核矩阵。`src/config.py`的通用CLI辅助函数不等同于`train.py`实际支持的参数。

---

## 7. 实验配置说明

### 7.1 核心配置项

核心配置定义在 [src/config.py:31-70](src/config.py#L31-L70)。

| 配置项 | 含义 | 默认值 |
|---|---|---|
| `experiment_name` | 实验名，用于输出文件命名 | `default_experiment` |
| `dataset_type` | 数据集编号 | `3` |
| `model_name` | 模型名 | `FusAtNet` |
| `cuda_device` | 运行设备 | `cuda:0` |
| `learning_rate` | 学习率 | `0.0001` |
| `epochs` | 训练轮数 | `10` |
| `batch_size` | Batch size | `128` |
| `num_workers` | DataLoader worker 数 | `0` |
| `random_seed` | 随机种子 | `6` |
| `channels` | PCA 后保留通道数 | `30` |
| `window_size` | 邻域窗口大小 | `11` |
| `enable_training` | 是否训练 | `True` |
| `enable_testing` | 是否测试 | `True` |
| `enable_visualization` | 是否生成分类图 | `True` |
| `enable_tsne` | 是否执行 t-SNE | `False` |
| `enable_erf` | 是否执行 ERF | `False` |

### 7.2 配置文件书写建议

`configs/train.json` 这类配置文件建议只保存“实验可变参数”，不要把框架能够自动推导的信息重复写进去。

推荐保留的字段通常包括：

- `experiment_name`
- `dataset_type`
- `model_name`
- `cuda_device`
- `learning_rate`
- `epochs`
- `batch_size`
- `num_workers`
- `random_seed`
- `channels`
- `window_size`
- `enable_training`
- `enable_testing`
- `enable_visualization`
- `enable_tsne`
- `enable_erf`

不建议手写的字段包括：

- `out_features`
- `data_channels`
- `lidar_or_sar_channels`
- `depth`（除非你确实要改默认结构）
- `model_save_path`
- `log_path`
- `report_path`
- `image_path`
- `image_path_web`

原因是这些字段会在 `ExperimentConfig` 初始化时，根据当前项目路径、数据集配置和实验参数自动补齐。尤其是各种输出路径，跨机器时不应该写死为绝对路径。

一个推荐的最小训练配置示例如下：

```json
{
  "experiment_name": "train_experiment",
  "dataset_type": 4,
  "model_name": "RSCNet",
  "cuda_device": "cuda:0",
  "learning_rate": 0.0001,
  "epochs": 100,
  "batch_size": 128,
  "num_workers": 0,
  "random_seed": 6,
  "channels": 30,
  "window_size": 11,
  "enable_training": true,
  "enable_testing": true,
  "enable_visualization": true,
  "enable_tsne": false,
  "enable_erf": false
}
```

### 7.3 输出文件命名规则

输出路径由 [src/config.py:97-122](src/config.py#L97-L122) 自动生成，目录按模型区分：

- `model/<ModelName>/...pth`
- `log/<ModelName>/...txt`
- `report/<ModelName>/...txt`
- `pic/<ModelName>/...png`

默认后缀形式为：

```text
_pca={channels}_window={window_size}_lr={learning_rate}_epochs={epochs}
```

如果指定 `experiment_name != default_experiment`，则会额外加上：

```text
_{experiment_name}
```

示例：

```text
model/RSCNet/Augsburg_model_train_experiment_pca=30_window=11_lr=0.0001_epochs=100.pth
log/RSCNet/Augsburg_log_train_experiment_pca=30_window=11_lr=0.0001_epochs=100.txt
report/RSCNet/Augsburg_report_train_experiment_pca=30_window=11_lr=0.0001_epochs=100.txt
pic/RSCNet/Augsburg_train_experiment_pca=30_window=11_lr=0.0001_epochs=100.png
```

### 7.4 推荐实验起步配置

#### 科研实验准备

先完成合成张量接口检查并批准组别与独立配置；正式实验固定100轮且至少seed6/7/8，不以1轮数据试训替代审核。下面的单配置只表示结构示例，不代表完整三种子矩阵。

#### 正式单模型实验

```python
ExperimentConfig(
    dataset_type=4,
    model_name="RSCNet",
    learning_rate=1e-4,
    epochs=100,
    batch_size=128,
    channels=30,
    window_size=11,
    enable_visualization=True,
    enable_testing=True,
)
```

#### 批量对比实验

建议使用：

```bash
python scripts/train_all.py --models RSCNet FusAtNet ExViT --datasets 0 4 5 --epochs 100
```

---

## 8. 输出结果说明

### 8.1 模型权重

训练器按既定评估轮次，在**测试OA提升**时保存最佳整模型；不存在独立验证集选模。最终报告沿用该checkpoint，不能描述为独立测试：

- 输出目录：`model/<模型名>/`
- 保存逻辑见 [src/trainer.py:97-100](src/trainer.py#L97-L100)

### 8.2 日志文件

训练日志由 [src/trainer.py:15-23](src/trainer.py#L15-L23) 写入，包含：

- 当前实验配置
- 训练开始时间 / 结束时间
- 每个 epoch 的 loss
- 每个 epoch 的测试集精度
- 最佳精度

### 8.3 测试报告

分类报告由 [src/metrics.py](src/metrics.py) 输出，包含：

- OA（Overall Accuracy）
- AA（Average Accuracy）
- Kappa
- 每类精度
- `classification_report`
- confusion matrix

### 8.4 可视化结果

分类图生成逻辑在 [src/analysis.py](src/analysis.py)。

不同数据集已内置：
- 图像尺寸
- 类别颜色映射

生成图片默认保存在：

```text
pic/<ModelName>/<Dataset>...png
```

### 8.5 t-SNE / ERF

- `t-SNE`：由 `enable_tsne` 控制，调用 [src/evaluator.py:64-65](src/evaluator.py#L64-L65)
- `ERF`：配置项中已预留 `enable_erf`，并存在 [src/show_erf.py](src/show_erf.py)、[src/visualize_erf.py](src/visualize_erf.py) 相关代码，但当前主流程集成度相对较低，建议按需单独接入

---

## 9. 开发者指南

### 9.1 如何新增一个模型

推荐按照现有模型目录结构新增：

```text
src/models/my_model/
├── adapter.py
├── model.py
└── __init__.py
```

#### 第一步：实现网络结构
在 `model.py` 中实现你的 PyTorch 模型类。

#### 第二步：实现适配器
参考 [src/models/fusatnet/adapter.py](src/models/fusatnet/adapter.py) 或 [src/models/rscnet/adapter.py](src/models/rscnet/adapter.py)，至少实现以下接口：

```python
def build_model(config, dataset_type, device):
    ...
    return {"net": net}


def forward_train(bundle, batch):
    ...
    return logits


def forward_eval(bundle, batch):
    ...
    return logits
```

`batch` 中可直接使用的字段通常有：
- `batch["hsi_pca"]`
- `batch["hsi"]`
- `batch["aux"]`
- `batch["label"]`

如果你的模型只用 PCA-HSI + 辅助模态，可参考 FusAtNet；
如果同时使用原始 HSI、PCA-HSI 和辅助模态，可参考 RSCNet。

#### 第三步：注册模型
编辑 [src/model_registry.py](src/model_registry.py)，加入：

```python
MODEL_REGISTRY = {
    ...,
    "MyModel": "src.models.my_model.adapter",
}
```

#### 第四步：先检查接口，再审核正式实验

先用合成张量核对形状、有限梯度、参数量和整模型保存加载，已有示例为`python -m unittest src.models.transmixer_sr.test_sr -v`。模型接口检查不产生科研成绩。新实验必须先批准组别与配置，再通过原入口按100轮及至少seed6/7/8执行；不自行增加1轮科研数据试训或改变训练协议。

### 9.2 如何新增一个数据集

新增数据集时，至少需要修改以下部分：

1. 在 [src/config.py](src/config.py) 中补充：
   - `DATASET_LABELS`
   - `out_features`
   - `data_channels`
   - `lidar_or_sar_channels`

2. 在 [src/data.py](src/data.py) 中补充：
   - 数据文件路径构造
   - `.mat` key 映射
   - `get<DatasetName>Data(...)`
   - `getMyData(...)` 中的数据集分发逻辑

3. 在 [src/metrics.py](src/metrics.py) 中补充：
   - 类别名称列表
   - `get<DatasetName>Report(...)`
   - `getMyReport(...)` 分发

4. 在 [src/analysis.py](src/analysis.py) 中补充：
   - 颜色映射
   - 图像尺寸
   - `vis<DatasetName>(...)`
   - `getMyVisualization(...)` 分发

5. 如果使用批量训练脚本，还需要更新 [scripts/train_all.py](scripts/train_all.py) 中的 `DATASET_REQUIRED_FILES`

### 9.3 如何新增命令行入口

如果你不想继续手改 `scripts/train.py` / `scripts/test.py` 里的配置，推荐新建一个更通用的入口，例如：

```python
from src.config import create_experiment_config_from_cli
from src.runner import ExperimentRunner

if __name__ == "__main__":
    config = create_experiment_config_from_cli()
    runner = ExperimentRunner(config)
    runner.print_config()
    runner.run_experiment()
```

这样就能直接使用统一 CLI 参数，无需每次编辑脚本文件。

### 9.4 开发建议

- 优先通过 `ExperimentConfig` 注入参数，不要回到散落的硬编码方式
- 模型差异尽量放在 `adapter.py` 中吸收，避免污染统一训练器
- 新增数据集时同步补齐报告与可视化，不然主流程虽然能跑，但结果不完整
- 正式实验前先完成合成接口与配置检查；科研数据上的新训练需按批准矩阵执行

---

## 10. 常见问题

### 10.1 找不到数据文件

先确认数据目录是否放在工作区同级 `data/` 下，而不是只放在仓库内。

可快速检查：

```bash
python -c "from src.data import DATA_ROOT; print(DATA_ROOT)"
```

### 10.2 GPU 不可用

检查：

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

如果不可用：
- 检查 NVIDIA 驱动是否正确安装
- 检查当前 Torch 是否为 GPU 版本
- 检查环境是否与 CUDA 12.1 / cu121 匹配

### 10.3 显存不足

可以优先降低：
- `batch_size`
- `window_size`
- `channels`

例如：

```python
ExperimentConfig(
    batch_size=32,
    window_size=9,
    channels=15,
)
```

### 10.4 训练可以跑但测试加载失败

测试阶段通过 [src/evaluator.py:55](src/evaluator.py#L55) 直接 `torch.load(model_savepath)` 加载权重，因此要确认：
- 对应模型文件已经生成
- 测试配置与训练配置一致
- 输出路径未因 `experiment_name`、`channels`、`window_size`、`lr`、`epochs` 不一致而改变

---

## 11. 建议的首次使用流程

如果你第一次接手这个项目，建议按下面顺序：

1. 阅读[项目导航](docs/PROJECT_GUIDE.md)、[最新研究总结（更新至2026-11-09）](docs/RESEARCH_PROGRESS_20261105.md)与[代码复用规则](AGENTS.md)。
2. 核对批次实际环境和本机数据目录，保留现有源码／数据／结果。
3. 执行`python scripts/verify_fusion_archives.py`检查已归档证据；接口检查使用合成张量。
4. 如需新实验，先展示组别、变量、配置与数量供审核，至少seed6/7/8。
5. 审核后用`python scripts/train.py --config <新配置>`运行，禁止覆盖或补跑已完成批次。
6. 使用原生报告归档结果，注明原测试OA选模限制。

---

## 12. 代码参考入口

若需要进一步阅读源码，建议从以下文件开始：

- [src/config.py](src/config.py)：先理解配置系统
- [src/runner.py](src/runner.py)：再看主流程如何组织
- [src/data.py](src/data.py)：理解数据格式和输入张量
- [src/trainer.py](src/trainer.py)：理解训练保存逻辑
- [src/evaluator.py](src/evaluator.py)：理解测试与可视化逻辑
- [src/model_registry.py](src/model_registry.py)：理解模型如何被统一接入

---
