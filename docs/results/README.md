# 实验结果导航

更新：2026-11-06。新完成[轻量化HAMF多尺度融合验证](fusion_hamf_seed678_20261105/README.md)、[第三批三篇新论文融合迁移](fusion_transfer_v3_seed678_20261104/README.md)与[局部／全局与SR机制分析](sr_lg_mechanisms_20261104/README.md)；阶段汇总见[11月4—6日研究记录](../RESEARCH_PROGRESS_20261105.md)和[11月2—3日研究总结](../RESEARCH_PROGRESS_20261103.md)。指标来自实验室原生报告；不同协议不能混算增益。

| 批次／快照 | 状态与数量 | 口径 |
|---|---|---|
| [轻量化HAMF多尺度融合（M00–M05）](fusion_hamf_seed678_20261105/README.md) | 24/24＝6复用＋18新训 | M00–M05，seed6/7/8；含开销审计、逐类表与尺度门控统计 |
| [第三批融合子机制迁移（CF-IPT/ISFuse/IID）](fusion_transfer_v3_seed678_20261104/README.md) | 39/39＝3复用＋36新训 | C01–C04/S01–S04/I01–I04，seed6/7/8；含开销审计与逐类表 |
| [局部／全局与SR机制研究](sr_lg_mechanisms_20261104/README.md) | 15个历史模型只读诊断＋6次新训；LG三种子复用 | LL／GG各seed6/7/8；含类别与邻域分析、冻结干预及完整核验 |
| [两日去重CSV](fusion_20261102_03_results.csv) | 45项＝3基线＋42新训 | seed6/7/8；OA/AA/Kappa及同种子OA差 |
| [三篇机制seed6](fusion_transfer_seed6_20261102/README.md) | 13/13＝1复用＋12新训 | F/U/D首轮，后被补种子汇总复用 |
| [三篇机制三种子完整汇总](fusion_transfer_seed78_20261102/README.md) | 39/39＝15复用＋24新训 | F01–F04/U01–U04/D01–D04，含首轮seed6 |
| [SR先验协同](sr_lg_prior_seed678_20261103/README.md) | 9/9＝3复用＋6新训 | A00/P01/P02，seed6/7/8 |
| [B1最终快照，11月3日补归档](fusion_B1_seed78_20261101_final_20261103/README.md) | 18/18＝10复用＋8新训；11月1日已完成 | G00–G05，seed6/7/8；不计入两日42新训 |
| [B1早期上传快照](fusion_B1_seed78_20261101/README.md) | 历史10/18进度，原样保留 | 不再代表最终结果 |
| [第一轮三篇融合seed6](fusion_papers_seed6_20261031/README.md) | 15/15＝2复用＋13新训 | B1/B2/B3筛查；B2/B3仅seed6 |
| [SR原流程](sr_native_20261029/README.md) | 27项 | 实验室原流程，含R/L/SR对照 |
| [旧论文迁移](paper_20261028/README.md) | 27项历史结果 | 旧验证选模，不与原流程混算 |
| [周末实验](weekend_20261024_25/README.md) | 历史多批次 | 依各批次协议分别解释 |

最新四份快照中的`SNAPSHOT.json`记录导出文件字节哈希，`COMPLETION_EVIDENCE.json`保存新训结束证据与原产物哈希，`native_reports/`为原生报告。完整日志和checkpoint留在本机。执行`python scripts/verify_fusion_archives.py`可离线复查快照及合并算术，不会训练或评估模型。

目录内的运行状态、报告和审核状态是快照当时事实；不回写旧快照。两日索引为[fusion_20261102_03_index.json](fusion_20261102_03_index.json)，合并CSV按组别＋seed去重。
