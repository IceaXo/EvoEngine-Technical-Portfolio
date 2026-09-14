# 资源引用与长上下文治理

长任务同时面临大结果与长历史。本人推进资源引用、历史压缩和文件上下文治理，使语义摘要与恢复需要的精确身份各自保留。

`messages` 是协议消息序列；`ResourceRef` 表达资源访问入口与元数据；`mechanical_state` 保存由记录提取的调用、资源、引用和副作用回执；语义摘要用于保留已知目标和进展。

1. 完整的大结果通过资源入口保留，模型按需要读取窗口或投影。
2. `compose_model_messages` 在这份冻结版本中保留消息，只计算 Token 等遥测信息；它本身不执行截断压缩。
3. 压缩规划选择可以处理的历史范围，精确状态由消息机械提取，再配合语义摘要形成 Checkpoint。
4. 文件登记维护引用和来源，供后续读取与交付校验使用。

阅读时重点对比 `extract_mechanical_state` 和语义压缩路径：call_id、资源 ID、任务完成回执不能由摘要模型编造。完成回执有独立的权威来源约束。

测试材料涵盖长消息、压缩状态和文件登记；本轮未实际计算模型 Token、压缩收益或成功率，不给出百分比。

## 代码阅读顺序

1. [ResourceRef](../source/agent/src/capabilities/models.py#L655)
2. [compose_model_messages](../source/agent/src/services/context_composer.py#L90)
3. [extract_mechanical_state](../source/agent/src/services/session_compaction.py#L653)
4. [plan_session_compaction](../source/agent/src/services/session_compaction.py#L799)
5. [context_composer_middleware.py](../source/agent/src/agents/context_composer_middleware.py)
6. [conversation_file_registry.py](../source/agent/src/services/conversation_file_registry.py)
7. [test_context_composer.py](../source/agent/tests/test_context_composer.py)
8. [test_session_compaction.py](../source/agent/tests/test_session_compaction.py)

[返回总览](../README.md) · [来源与整理边界](PROVENANCE.md) · [依赖与验证](DEPENDENCIES.md)
