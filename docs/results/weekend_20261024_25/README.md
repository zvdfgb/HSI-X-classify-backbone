# TransMixer 周末研究归档

实验日期：2026-10-24 至 2026-10-25。归档日期：2026-10-26。

本轮围绕固定局部／全局 1∶1 的 TransMixer，验证跨模态消息交换与教师指导是否能改善 Houston2013 的 HSI＋LiDAR 分类。共完成 **31 次正式训练**，分成两套评价协议。没有新增 Berlin／SAR 数据实验，没有把试启动、初筛和接口测试算作正式结果。

## 先看结论

- 原流程 F0 直接融合 OA 为 88.8907%，接近历史 88.9727%。
- 同批 F1 为 89.0957%，F4 双向交互为 89.2761%；F4−F0 为 +0.3853，F4−F1 为 +0.1804 个百分点。交互只有小幅正向信号，尚未证明独立机制收益。
- 新增四组蒸馏均低于各自未蒸馏学生。RSCNet 教师 OA 为 91.6373%，但 F4 向其蒸馏后只有 89.0547%。本轮固定蒸馏方案没有提升，不代表所有蒸馏方法无效。
- F4 标签平滑达到 89.4072%，相对 F4 +0.1312 个百分点；单次小差值不能认定稳定有效。
- 置信过滤阈值 0.7 接受了全部训练样本，没有真正形成过滤；不能据此否定可靠性过滤假设。

## 报告与数据

- [中文完整报告 Markdown](周末研究报告.md)
- [可编辑 Word](TransMixer%20周末融合与知识蒸馏实验全面报告.docx)
- [PDF 报告](TransMixer%20周末融合与知识蒸馏实验全面报告.pdf)
- [原流程十组指标 CSV](原流程十组结果.csv)
- [完整逐类召回率 CSV](逐类召回率.csv)
- [验证选模三种子统计](validation_fusion/正式汇总.csv)
- [原始证据仓库索引](evidence/index.json)
- [封存及报告验收](最终封存复核.json)
- [归档文件指纹](archive_manifest.json)

## 两套协议必须分开

| 批次 | 训练次数 | 设置 | 选模规则 |
|---|---:|---|---|
| F0 至 F6 融合对照 | 21 | seed6/7/8；2267 训练、565 验证、12197 测试；100 轮 | 验证 OA，同分取较低验证 CE |
| 原流程 F0 F1 F4 | 3 | seed6；官方 2832 训练、12197 测试；100 轮 | 原实验室测试 OA 最佳 checkpoint |
| 两位教师及五组学生对照 | 7 | 同上；温度4、KD系数0.3、置信阈值0.7、标签平滑0.1 | 同上 |

共同使用 PCA30、11×11 窗口、batch128、AdamW lr=0.0001、weight_decay=0.01。原流程 drop_last=True，每轮22批。全场景无标签 PCA／标准化属于传导式预处理。

原流程结果不是验证选模后的独立测试。验证批次的空间邻域存在大量重叠、验证 OA 饱和；两批都未完成空间隔离的泛化确认。不能混合排名、宣称统计显著或 SOTA。RSCNet 额外使用原始144波段 HSI，因此教师对比还涉及输入信息量差异。

## 实现与复现边界

- 融合实现：`src/models/transmixer_fusion/`；七组独立验证选模入口：`scripts/train_transmixer_fusion.py`，配置 `configs/fusion_confirm.json`。详见 [融合实验指南](../../transmixer_fusion_experiments.md)。
- 蒸馏实现：`src/models/transmixer_distill/`；原 `src/trainer.py` 仅添加可选损失和结束诊断回调，默认 CE、优化器、数据及 checkpoint 规则保留。
- 原流程实际入口是已有 `scripts/train.py --config <配置>`。每个学生由原 B0 或 F4 从头初始化，教师冻结，只在训练时参与，不进入学生 checkpoint。
- `evidence/native_reports/` 保存本轮十份原生标准报告；`evidence/native_configs/` 保存当时的完整配置；`validation_fusion/jobs/` 保存21份结果和100轮历史。
- 模型权重、原始数据集、验证逐样本预测及大体积过程日志仍在本地 `model/`、`data/`、`report/`，未进入 Git。本归档支持审阅已保存结果，不包含无需数据与权重即可重新推理的完整环境。
- `scripts/train_transmixer_distillation.py` 和 `configs/weekend_distillation.json` 保留当时“等待旧批次、封存旧源码、接入蒸馏”的一次性调度流程，依赖旧批次目录与当时源码阶段。**当前仓库已接入蒸馏，不应直接运行该历史迁移队列。** 新实验通过原训练入口加载自行设置新 experiment_name 的配置；蒸馏配置需提供本地教师 checkpoint 路径、其 SHA256 和诊断输出路径。
- `scripts/compare_lab_fusion.py` 会核对历史源码指纹，不能在已接入蒸馏的源码上对旧批次直接重汇总；请读取本目录已封存的旧批次汇总。
- Word/PDF 是当时已逐页核验的原始交付件；Markdown 仅将图片链接转为仓库相对路径。文档再生成工具包含本机 Windows Word／Codex runtime 路径，换机器时需先调整。既有报告阅读不依赖这些工具。
- 作业调度依赖 `psutil`，已补入训练依赖清单。报告再生成另需 `python-docx`、`pypdf`、`pdf2image`、Word 或 LibreOffice 及 Poppler；本机通过预装文档运行环境完成。

原始 manifest、CSV 中保留的本机绝对路径是来源记录；仓库可浏览的对应文件以 `evidence/index.json` 为准。本归档通过 `.gitattributes` 禁止换行转换，可按 `archive_manifest.json` 复核归档文件字节。历史源码指纹针对实验时的本地原始字节；主代码目录的 Git 换行转换、后续代码修改或报告相对路径转换会改变字节，不应当作训练期间修改的证据。

## 提交前检查

```powershell
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
python -m unittest src.models.transmixer_fusion.test_fusion src.models.transmixer_fusion.test_protocol src.models.transmixer_distill.test_distillation src.models.transmixer_ablation.test_feature_ablation -v
```

这些检查覆盖融合方向、初始化、梯度、划分、确定性、教师冻结、零蒸馏权重、保存恢复与旧接口兼容；其中合成数据的短程检查不算正式训练。本次同步不重复运行科研训练或测试集推理。
