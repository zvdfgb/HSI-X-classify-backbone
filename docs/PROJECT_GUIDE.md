# 项目训练与目录导航

更新：2026-10-29。当前研究方向：固定1:1局部／全局通道分工＋SR2邻域加权汇总。新实验等待审核，上轮27项保留、不重跑。

## 日常只看这些位置

| 位置 | 用途 |
|---|---|
| `configs/train.json` | 实验室默认训练标准，保持原文件 |
| `scripts/train.py`、`scripts/test.py` | 单次训练／测试入口 |
| `scripts/train_all.py`、`scripts/run_batch.py` | 实验室已有批量入口，显式指定所需模型 |
| `src/data.py` | 唯一数据加载、PCA、归一化、增强与官方划分实现 |
| `src/trainer.py`、`src/evaluator.py` | 唯一训练、选模、测试实现 |
| `src/models/transmixer_net/` | 原骨干、固定1:1和纯SSM模式 |
| `src/models/transmixer_sr/adapter.py` | SR1／SR2接入原流程，只有模型设置和前向接口 |
| `src/model_registry.py` | 当前模型与历史兼容模型分开登记 |
| `docs/experiments/` | 待审核／已审核的实验设计；执行前核对状态 |
| `docs/results/` | 已封存的结果证据，按批次查阅 |
| `docs/archive/assistant_experiments_20261029/` | 撤出日常入口的助手脚本、旧配置和失效方案 |
| `data/`、`model/`、`log/`、`report/`、`pic/` | 本机数据、权重、日志、报告和图像；不移动、不覆盖、不纳入Git |

## 后续训练流程

1. 审核实验组与固定参数。
2. 复制原 `configs/train.json` 形成每组每种子的配置；只修改明确列出的模型设置、种子与实验名称。原训练参数不变。
3. SR2的距离尺度通过原 `getMyData` 返回的训练 Dataset 输入估计，估计过程独立执行，不读取测试标签或重写PCA。相同种子下SR2各组共用一个尺度并记录来源，禁止照搬旧验证划分的数值。此准备步骤待实验审核后落实。
4. 使用原入口运行，例如 `python scripts/train.py --config <已审核的配置文件>`。不要直接运行未指定模型的全模型队列：SR2要求显式提供模型尺度；历史兼容模型不在默认队列中。
5. 原 `src/evaluator.py` 生成OA、AA、Kappa等报告；额外汇总只读取这些结果，不另写预测／指标流程。

SR模型名为 `TransMixer_SR1`、`TransMixer_SR2`；`ablation_mode` 支持现成的 `decoupled`（固定1:1）和 `pure_ssm`（无局部／全局分工）。配置字段 `sr` 仅包含 `spatial_sigma` 与 `spectral_scale`。SR1不需要光谱尺度；SR2缺少尺度会明确报错。模型直接复用上一轮 `NeighborTransMixer` 和实验室 `TransMixerNet`，没有复制网络主体。

## 原实验室规则

Houston2013官方2832训练／12197测试，不再切出验证集。PCA30，11×11反射补边；HSI／辅助模态归一化、同步翻转全按 `src/data.py`。batch128，训练 `drop_last=True`，workers0；100轮，AdamW，lr=0.0001，weight_decay=0.01。随机性与cuDNN设置完全由原代码执行，不额外强制另一套确定性开关。

原训练器在第1、10、20、…、100轮以及85、95轮评估，用测试OA保存最好整模型。后续比较沿用这条规则，并在结果中标明“实验室原流程”，不称作验证选模后的独立测试。用户未批准前不改变这条规则。

## 旧实验与兼容代码

- [paper_20261028结果](results/paper_20261028/README.md)：27项已完成，验证选模口径。SR2相对G0平均提高3.3342个百分点，作为继续研究的依据。
- [周末归档](results/weekend_20261024_25/README.md)：包含不同口径的结果，分表查阅。原流程固定1:1基线seed6为88.8907%；不能用它与旧验证流程85.3270%均值直接算增益。
- `transmixer_paper`、`transmixer_fusion`、`transmixer_ablation`、`transmixer_distill`、`transmixer_deg` 中的历史类保留原路径，防止旧整模型权重加载失效。PN／OC／SR3等仅作兼容，不进入新研究队列。SR1／SR2复用的旧模型类仍是当前依赖。
- 已撤出的27个文件及原始哈希见归档 `archive_manifest.json`。历史脚本还原运行应检出对应历史提交，不能直接在归档目录执行。
- `protected_core_sha256.json` 记录整理前13个核心文件的哈希；此次整理后逐一比对，验证实验室核心没有被改写。

合成接口检查命令：`python -m unittest src.models.transmixer_sr.test_sr -v`。它只检查模型接口、梯度、参数一致与整模型保存加载，不读取科研数据、不生成实验成绩。
