# SR与局部/全局分工协同验证

## Goal
执行用户已批准的正确SR注意力先验与错位先验两组，各seed6/7/8、100轮；复用原A三个种子，共9项（新训6）。

## Context From Memory
默认docs/knowledge-base不存在，无索引条目。约束直接来自本聊天、AGENTS.md及已核验旧manifest，不把历史记忆当运行证据。

## Scope and constraints
仅新增标准adapter模型、实验配置、编排/核验与临时防休眠脚本，以及注册表两项接入。复用原SSM、固定1:1分工、MHA、局部卷积、模态融合与SR2；原训练器/数据/损失/优化器/选模不改。旧源码与产物只读，注册表修改前原字节与哈希归档。禁止额外组别、种子、调参、早停、静默重训。

## Assumptions
每个现有block一个零初始化无约束标量beta，训练器默认AdamW共同优化；错位使用固定无不动点的121位置置换（本地Generator seed20261103），所有训练种子共用，不消耗全局随机状态。只打乱注意力先验，最终SR读出保持正确位置。

## Acceptance criteria
- beta为0且共享参数一致时，新模型与原A前向结果在数值容差内一致，P01/P02完全配对；新增标量外原初始化与随机状态不变。
- 非零beta与原MultiheadAttention加显式加性mask结果一致，错位保留权重分布但破坏位置对应。
- 原生batch128前向/反向与参数梯度有限；beta梯度非零；整模型checkpoint可保存加载。
- 两新组参数量相同；所有配置遵守100轮/seed6-8及原协议；A三份封存报告与checkpoint核验复用。
- 每项完成须有第100轮22/22、Report Success及五类产物哈希，汇总最终OA/AA/Kappa与三种子样本标准差、逐种子配对差值。

## Edge cases and guardrails
预检失败不启动；发现活动训练或目标产物已存在则拒绝启动/覆盖。训练失败停止队列，不自动重训；保留活动子进程证据。临时防休眠守护覆盖队列及仍存活子进程，全部结束释放，不改电源计划、不设置关机。

## Validation and execution
归档旧锁定证据→模型与注册表接入→接口/数值/梯度/序列化预检→生成6配置与9任务manifest锁定→隐藏队列进程调用原scripts/train.py→小时监控→最终日志与哈希复核。

## Self-check
Ready for implementation。无阻塞问题。用户已授权执行；本规格未扩展实验矩阵。
