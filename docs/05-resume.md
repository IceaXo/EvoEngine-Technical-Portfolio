# Checkpoint、异步作业与授权恢复

等待计算或人工确认时，系统需要在原请求上恢复。本人推进 Checkpoint、异步续接、HITL 与嵌套作用域问题的诊断和集成。

先认清四个身份：`request_id` 是原会话轮次，`call_id` 是原调用，`job_id` 是已提交作业，授权记录约束原用户与原请求。`checkpoint_id` 标识保存的快照，不能代替上述身份。

1. 协议账本闭合后，构建包含目标、消息、工具装载状态、待完成作业、资源和授权控制的 TurnCheckpoint。
2. 服务端按 project_session_id/request_id 找到 `ProjectConversationTurn`，将快照写入 `thinking_trace_json["turn_checkpoint"]` 并提交数据库。
3. 作业终态通知触发续接；服务端核对用户、会话、原 request 和待处理 job，读取原快照，构造 continuation。
4. Agent 恢复消息和原执行状态，继续决策或原待处理调用；完成通知本身不等于任务已经交付。

Worker 协议先持久化接受记录再启动线程；同一 job_id 的不同参数会冲突。重启后，原 pending/running 作业被标记失败，避免在旧进程状态未知时自动重跑。**这是该状态存储保留期内的 at-most-once 策略，终态记录会清理；不是跨所有实例与无限时间的 exactly-once。** 同一状态目录的跨进程并发写入也不能由线程锁保证。

超时更新入口先验证用户拥有该作业，只允许活动状态和合法秒数。模型流停滞、作业运行超时、HITL 等待分别有自己的状态，不应用一个笼统重试覆盖。

原数据库恢复用例会清空进程内记录，再按相同请求读回 Checkpoint；它证明的测试意图是持久化权威路径，不能单独证明完整科研任务在崩溃后自动完成。回调续接、嵌套授权和 Worker 测试另行保留，本轮均未执行。

## 代码阅读顺序

1. [_build_turn_checkpoint](../source/agent/src/agents/native_agent_graph.py#L4008)
2. [_restore_runtime_from_turn_checkpoint](../source/agent/src/agents/native_agent_graph.py#L4087)
3. [_resume_pending_capability_call](../source/agent/src/agents/native_agent_graph.py#L4214)
4. [_persist_turn_checkpoint_payload](../source/server/backend/server_backend.excerpt.py#L326)
5. [_load_turn_checkpoint_payload](../source/server/backend/server_backend.excerpt.py#L358)
6. [_continue_sandbox_turn](../source/server/backend/server_backend.excerpt.py#L862)
7. [ProjectConversationTurn](../source/server/backend/database.excerpt.py#L89)
8. [turn_authorization_service.py](../source/server/backend/services/turn_authorization_service.py)
9. [durable_job_protocol.py](../source/server/sandbox_workers/durable_job_protocol.py)
10. [update_job_timeout](../source/server/backend/services/sandbox_service.excerpt.py#L157)
11. [test_hitl_resume_flow.py](../source/server/backend/tests/test_hitl_resume_flow.excerpt.py)
12. [test_durable_job_protocol.py](../source/server/sandbox_workers/test_durable_job_protocol.py)

[返回总览](../README.md) · [来源与整理边界](PROVENANCE.md) · [依赖与验证](DEPENDENCIES.md)
