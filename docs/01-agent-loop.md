# Agent 主链与事件流

用户目标需要经过多轮工具调用才能得到结果。本人负责主链搭建、能力接入和执行过程集成；开发使用 AI 辅助，方案取舍、集成与验证属于本人工作。

输入先看 `NativeAgentState`：当前消息、目标、已加载工具、结果与恢复信息；`NativeAgentContext` 携带运行依赖。`build_dynamic_agent` 中的 `llm` 是模型，`tools` 是本轮开放的工具集合，`middleware` 负责横切控制。不能把 `create_agent` 本身说成自行实现的模型推理框架。

1. 请求转换成运行状态，按历史或 Checkpoint 恢复已有结果。
2. `build_dynamic_agent` 组装模型、工具与中间件，再调用 LangChain `create_agent`。
3. `_run_native_agent_turn_once` 消费模型/工具事件，更新调用协议、工具结果与运行状态；等待、恢复、终止分别走自己的分支。
4. 事件流向外传递中间进度和终态。工具执行成功后还需要模型继续消费结果，才能判断任务下一步。

原有迭代驱动测试关注驱动形式和连续执行结构。本仓库保留完整 native 驱动模块，以及 lead_agent 中完整的组装、完成工具和事件流函数。没有重写成一个缩小版 Agent。

阅读边界：模型提供商、服务端接口和部分中间件仍来自完整工程。文件存在与 AST 可解析不能证明服务能启动或长任务通过验收。

## 代码阅读顺序

1. [build_dynamic_agent](../source/agent/src/agents/lead_agent.excerpt.py#L503)
2. [_run_native_agent_turn_once](../source/agent/src/agents/native_agent_graph.py#L4392)
3. [stream_native_agent_events](../source/agent/src/agents/native_agent_graph.py#L8258)
4. [test_native_agent_iterative_driver.py](../source/agent/tests/test_native_agent_iterative_driver.py)

[返回总览](../README.md) · [来源与整理边界](PROVENANCE.md) · [依赖与验证](DEPENDENCIES.md)
