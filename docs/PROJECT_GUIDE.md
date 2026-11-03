# 项目训练与目录导航

更新：2026-11-03。当前主线为固定1:1局部／全局通道分工＋SR2。三篇融合机制三种子和SR先验协同批次均已完成，当前没有待继续的训练队列。最新判断见[两日研究总结](RESEARCH_PROGRESS_20261103.md)。

## 目录职责

| 位置 | 用途 |
|---|---|
| `configs/train.json` | 实验室默认标准，保留原文件 |
| `configs/runs/<批次>/` | 已批准的逐组逐种子配置；原文件保持封存，不原地覆盖复跑 |
| `scripts/train.py`、`scripts/test.py` | 单次原生训练／评估入口 |
| `src/runner.py`、`src/trainer.py` | 原实验调度、训练、AdamW与选模 |
| `src/data.py`、`src/evaluator.py`、`src/metrics.py` | 唯一科研数据处理、最终评估及指标实现 |
| `src/models/transmixer_net/` | 实验室骨干及固定1:1／纯SSM对照 |
| `src/models/transmixer_sr/` | SR1／SR2的标准adapter；复用NeighborTransMixer |
| `src/models/transmixer_fusion_papers/` | 首轮B1/B2/B3迁移与兼容代码 |
| `src/models/transmixer_fusion_v2/` | F/U/D机制模型及附加损失adapter |
| `src/models/transmixer_sr_prior/` | P01/P02：复用原全局注意力与SR，仅增加先验偏置 |
| `src/model_registry.py` | 当前模型与历史兼容模型注册 |
| [experiments/](experiments/README.md) | 审核矩阵、执行记录、迁移源码证据 |
| [results/](results/README.md) | 轻量、可核验的结果快照和合并表 |
| `scripts/run_fusion_transfer*.py`、`scripts/run_sr_lg_prior.py` | 已完成批次的封存、排队与审计控制器；训练仍调用原入口 |
| `scripts/verify_fusion_archives.py` | 仅校验Git内快照、原报告解析值及汇总算术 |
| `docs/archive/` | 只读历史追溯，不恢复为日常入口 |
| `data/`、`model/`、`log/`、`report/`、`pic/` | 本机数据与运行产物，不移动、不覆盖、不纳入Git |

## 新实验执行约定

1. 先展示并审核组别、变量、设置与数量。后续各组至少seed6/7/8，不自动增加模型、种子或调参。
2. 从`configs/train.json`派生独立配置，只改批准字段，实验名包含批次／组别／seed，禁止覆盖旧结果。
3. SR2光谱尺度由原`getMyData`训练Dataset输入统计，不读取测试标签，不重写PCA。本次已完成批次使用封存值0.2775005102157593、spatial_sigma=3；来源见批次配置／manifest。新设计需核对数据和已有统计的适用性。
4. 原入口为`python scripts/train.py --config <新批准配置>`。不要重跑已完成队列或直接调用无明确模型列表的全模型批量入口。
5. 新模型只通过adapter挂接；确有接口缺口先说明并做最小修改。原生报告产生后才能汇总，不另写预测／科研指标替代流程。

## 固定协议与解释限制

Houston2013官方2832训练／12197测试，无新增验证划分；PCA30、11×11反射补边、原归一化与同步翻转。batch128、drop_last=True、workers0；100轮、AdamW lr=1e-4、weight_decay=0.01。原随机数及cuDNN设置不额外更换。

原训练器在第1、10、20、…、100轮以及85、95轮使用测试OA保存最佳整模型。此规则未经单独授权不改，结果不称为独立测试。F/U/D中的附加损失是批准的实验变量；具体系数与梯度隔离见对应矩阵和源码。

## 只读复查与历史兼容

```bash
python scripts/verify_fusion_archives.py
python -m unittest src.models.transmixer_sr.test_sr src.models.transmixer_fusion_papers.test_fusion -v
```

第一条只验证仓库证据与算术，无需GPU／科研数据；第二条使用合成张量检查接口、梯度和保存加载，不训练科研实验。原机完整产物核验依照批次执行记录，调用批次运行器中的只读审计接口。`verify_fusion_transfer.py`／`verify_sr_lg_prior.py`是运行前接口预检工具，可能写入preflight.json，不能当作归档只读核验命令直接重跑；也不重跑launch。

历史整模型checkpoint依赖原类路径，`transmixer_paper`、`transmixer_fusion`、`transmixer_ablation`、`transmixer_distill`、`transmixer_deg`等保留。PN／OC／蒸馏等停止新增实验；被SR复用的旧类仍是当前依赖。旧验证选模结果与原流程成绩分开查阅，禁止混算增益；2026-10-28的27项不重跑。

本次整理只补导航与证据归档，不搬迁原模型／训练代码。实际批次环境在各结果快照`RUNTIME.json`，通用`environment.yml`不是这些批次的精确锁定环境。
