# 三篇论文迁移实验归档

实验标识：`paper_20261028`。完成时间：2026-10-29 02:35:22（北京时间）。

本目录保存 Git 可读的结果与来源证据。共9组、每组种子6/7/8：3项复用并验证旧基线，24项新增训练，27项最终测试全部完成。完整训练日志、逐样本预测、检查点和源文件快照仍保留在本机 `report/paper_transfer/paper_20261028/`，不提交数据集及模型权重。

- `实验报告.md`：各组均值、标准差及配对差值。
- `summary.json`、`summary.csv`：逐组逐种子结果。
- `paired_comparisons.json`：完整配对比较，不删负结果。
- `manifest.json`、`protocol_check.json`：固定配置、来源哈希与环境。
- `training_input_statistics.json`：只从训练输入估计的距离及几何尺度。
- `运行状态.json`：完成状态。

共同标准来自 `report/fusion_ablation/fusion_20261024_210925_749584`；采用验证集选择检查点。全部固定配置训练完毕后统一执行最终测试。G0本身已经包括固定1:1局部／全局通道划分，因此SR2是“1:1＋SR”的组合，而不是单独SR。

SR2测试OA为88.6611±1.4534%，较G0平均提高3.3342个百分点，三个种子全部提升；相对SR1平均提高1.3528个百分点，三个种子全部提升。SR3没有在SR2之上提供稳定增益。PN和OC作为历史探索保留，后续研究按用户要求聚焦SR，不删除历史证据。

代码检查命令（使用与正式实验相同的Python环境，先设置确定性CUDA环境变量）：

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python -m unittest src.models.transmixer_paper.test_paper src.models.transmixer_fusion.test_protocol src.models.transmixer_fusion.test_fusion -q
```

审核稿与执行规范位于 `docs/experiments/20261028_三篇论文迁移实验_审核稿.md` 和 `docs/experiments/20261028_执行规范与代码复用.md`。审核稿记录执行前设计，以本目录完成报告为结果依据。
