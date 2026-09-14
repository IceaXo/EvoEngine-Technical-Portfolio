# 调用身份、结果账本与幂等

同一工具可以连续调用多次。只按工具名称填回结果，会把结果绑定到错误请求；重复到达的执行请求还可能重复产生副作用。本人参与协议收敛、错误定位与幂等链路集成。

先区分变量：`call_id` 标识一次模型调用；`arguments` 是该次输入；`idempotency_key` 标识允许复用的执行语义；`task_node_id` 约束任务动作；Provider 的 job/执行记录用于定位实际副作用。

1. 协议账本登记原始调用，校验结果的 call_id、工具名与规范化参数身份。
2. `commit_tool_result` 对相同结果可重复提交，对同一调用的冲突结果拒绝写入。
3. Executor 在动态任务节点投影后，必要时重建幂等键；已有记录先 restore，再决定是否执行。
4. 绑定任务的非幂等能力要求明确的 Provider reconcile 适配器；无法确认原 begin 身份时返回合同错误。

对应测试检查缺失调用、错误绑定、冲突结果和幂等身份。协议账本的“闭合”表示每个调用有对应结果；它不等于外部副作用恰好执行一次。Provider 的持久化范围、过期策略和恢复能力仍决定实际保证。

## 代码阅读顺序

1. [canonical_protocol_ledger.py](../source/agent/src/agents/canonical_protocol_ledger.py)
2. [ledger.py](../source/agent/src/capabilities/ledger.py)
3. [executor.py](../source/agent/src/capabilities/executor.py)
4. [tool_result_envelope.py](../source/agent/src/services/tool_result_envelope.py)
5. [test_canonical_protocol_ledger.py](../source/agent/tests/test_canonical_protocol_ledger.py)
6. [test_capability_idempotency.py](../source/agent/tests/test_capability_idempotency.py)
7. [test_capability_ledger.py](../source/agent/tests/test_capability_ledger.py)

[返回总览](../README.md) · [来源与整理边界](PROVENANCE.md) · [依赖与验证](DEPENDENCIES.md)
