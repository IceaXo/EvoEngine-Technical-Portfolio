# 统一能力合同、发现与 Provider

Skill、MCP、Sandbox、内部工具与子 Agent 的调用方式不同，但主链需要一致的参数、错误、资源和状态接口。本人推进统一能力层及多类能力集成。

`spec` 描述能力版本、输入 Schema、权限与执行策略；`call` 携带调用身份、参数、输入资源及任务绑定；`result` 表示输出、资源、错误和运行状态。`registry_snapshot_id` 用来确认调用所针对的注册快照。

1. 发现层提供候选能力信息；实际工具 Schema 按需要装载到主链。
2. Executor 校验注册快照、可用状态、参数与文件输入，定位对应 Provider。
3. 在执行前处理任务节点投影、绑定与已有执行记录；不满足合同的调用返回明确失败。
4. Provider 适配具体执行后端，再将结果投影回统一合同供上层继续处理。

目录保留所有已选 Provider 模块，能对照 MCP、Skill 和 Sandbox 的真实适配差异。相关模型、发现和 Executor 测试也保留在 `source/agent/tests`。

边界：注册表中存在某能力不代表其服务此刻可达；此处没有连接任何远端服务，也没有把所有 Provider 说成已在线验收。

## 代码阅读顺序

1. [CapabilitySpec](../source/agent/src/capabilities/models.py#L757)
2. [CapabilityCall](../source/agent/src/capabilities/models.py#L968)
3. [CapabilityResult](../source/agent/src/capabilities/models.py#L1092)
4. [discovery.py](../source/agent/src/capabilities/discovery.py)
5. [registry.py](../source/agent/src/capabilities/registry.py)
6. [executor.py](../source/agent/src/capabilities/executor.py)
7. [mcp.py](../source/agent/src/capabilities/providers/mcp.py)
8. [skill_script.py](../source/agent/src/capabilities/providers/skill_script.py)
9. [sandbox.py](../source/agent/src/capabilities/providers/sandbox.py)
10. [subagent.py](../source/agent/src/capabilities/providers/subagent.py)
11. [test_capability_executor.py](../source/agent/tests/test_capability_executor.py)

[返回总览](../README.md) · [来源与整理边界](PROVENANCE.md) · [依赖与验证](DEPENDENCIES.md)
