# 文件血缘、完成验收与可信回执

模型说“完成了”与系统确认交付是两件事。本人推进文件登记、血缘和服务端完成合同，防止任意字符串被当成可信产物或完成凭据。

输入包含任务树根、结果摘要、交付文件 ID 及登记来源；`receipt` 是服务端根据持久化事实生成的完成回执。文件引用需要对应真实登记记录，不能只给一个路径字符串。

1. 文件内容通过有作用域的登记入口写入并获得引用，保留来源信息。
2. Agent 完成工具提出结果；服务端锁定任务树记录，校验子节点、文件、状态等完成条件。
3. `complete_task_root` 在同一事务中更新根/运行状态并保存回执；重复完成请求必须与原结果身份一致。
4. `resolve_authoritative_completion_receipt` 按项目、会话与 request 查询持久化记录，并比较回执的规范内容；聊天终态消费该权威结果。

完成合同测试和回执解析测试分别覆盖上游验收与下游消费，二者不能只用一个解析函数代替。阅读版保留两端和调用位置。

边界：本仓库没有重跑数据库、浏览器或真实文件交付验收。原测试是验证依据与可审阅的断言，不是本轮测试通过报告；科研阴性也不应因“没有阳性结果”被自动算作工具失败。

## 代码阅读顺序

1. [bind_file_registration_authority](../source/agent/src/services/conversation_file_registry.py#L36)
2. [register_conversation_file_text](../source/agent/src/services/conversation_file_registry.py#L491)
3. [build_task_complete_tool](../source/agent/src/agents/lead_agent.excerpt.py#L125)
4. [_validate_tree_completion](../source/server/backend/services/task_tree_service.excerpt.py#L58)
5. [complete_task_root](../source/server/backend/services/task_tree_service.excerpt.py#L730)
6. [resolve_authoritative_completion_receipt](../source/server/backend/services/task_completion_receipt_service.py#L56)
7. [_finish_turn_success](../source/server/backend/server_backend.excerpt.py#L378)
8. [test_task_tree_root_completion_contract.py](../source/server/backend/tests/test_task_tree_root_completion_contract.py)
9. [test_task_completion_receipt_service.py](../source/server/backend/tests/test_task_completion_receipt_service.py)

[返回总览](../README.md) · [来源与整理边界](PROVENANCE.md) · [依赖与验证](DEPENDENCIES.md)
