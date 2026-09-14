# 依赖与运行边界

这是原工程的阅读选编，不能直接对根目录运行 pytest 或把节选模块当作可启动应用。没有为缺失依赖伪造空实现或提供会连接真实服务的一键脚本。

## 依赖分层

| 层 | 原工程依赖与未包含内容 |
| --- | --- |
| Agent 主链 | LangChain/LangGraph、模型适配、提示词与工具初始化、内部 src 服务模块 |
| 能力层 | Pydantic 合同、Provider 客户端、注册目录、授权和执行状态存储 |
| Server | FastAPI、SQLAlchemy、完整数据库模型、配置、会话服务及执行器生命周期 |
| 科研工具 | 文献和数据库提供商、MCP/Skill 服务、Sandbox 生信软件与数据 |
| 测试 | 原 conftest、完整内部模块、隔离数据库与 fixture；原测试有 monkeypatch/fake，不等于真实后端验收 |
| Worker | 持久化目录、runner、进程生命周期；实现含 POSIX `os.O_DIRECTORY`，不能据无第三方依赖推导 Windows 可运行 |

所有 Python 文件的实际 import 保存在 `source_manifest.json`，可逐项追溯。节选文件保留原 import 与原调用名；相应完整模块不一定存在于本仓库。

## 节选文件

| 文件 | 保留的完整函数或类 |
| --- | --- |
| `source/agent/src/agents/lead_agent.excerpt.py` | `TaskCompleteArgs`, `_task_complete_impl`, `build_task_complete_tool`, `build_dynamic_agent`, `stream_agent_events` |
| `source/server/backend/database.excerpt.py` | `ProjectTaskTreeRun`, `ProjectConversationTurn` |
| `source/server/backend/server_backend.excerpt.py` | `_persist_turn_checkpoint_payload`, `_load_turn_checkpoint_payload`, `_sandbox_turn_resume_snapshot`, `_wait_for_sandbox_turn_pause`, `_continue_sandbox_turn`, `_drain_sandbox_turn_continuations`, `_schedule_sandbox_turn_continuation`, `_terminal_pending_sandbox_payloads`, `_recover_sandbox_turn_continuations`, `_finish_turn_success`, `_finalize_turn_terminal` |
| `source/server/backend/services/sandbox_service.excerpt.py` | `build_job_terminal_callback_payload`, `get_job_or_404`, `_normalize_job_timeout_seconds`, `update_job_timeout` |
| `source/server/backend/services/task_tree_service.excerpt.py` | `_validate_tree_completion`, `_validated_completion_receipt`, `_build_completion_receipt`, `complete_task_root` |
| `source/server/backend/tests/test_hitl_resume_flow.excerpt.py` | `_bind_session_test_database`, `_ensure_test_tables`, `_ensure_user`, `_timeline_answer`, `_hitl_lineage`, `_same_request_task_authority`, `_create_completed_task_tree_receipt`, `test_pending_sandbox_job_resume_state_is_not_silently_capped`, `test_turn_checkpoint_is_database_authority_after_runtime_state_is_gone`, `test_hitl_continuation_keeps_full_parent_user_request`, `test_build_agent_http_payload_keeps_identity_when_custom_payload_present`, `test_hitl_parent_scope_resolves_original_conversation`, `test_stream_from_agent_http_keeps_waiting_sandbox_non_terminal`, `test_sandbox_terminal_callback_runs_agent_before_finalizing`, `test_sandbox_finalize_callback_includes_turn_owner`, `test_sandbox_terminal_callbacks_are_serialized_per_turn`, `test_hitl_lineage_mints_exact_parent_run_and_nested_authority` |

函数体中的日志、全局状态、常量与辅助调用来自原模块。完整项目依赖和配置尚未被移植到这里；AST 解析通过只表示 Python 语法成立。

原测试保留用于阅读验证意图。在获得完整依赖、独立测试环境及运行授权前，本次不执行它们。后续若构建可运行的独立组件，应单独标记新版本和改动，不能声称就是冻结工程的全量复现。
