# 检索子 Agent、来源身份与 EvidencePack

科研结果需要从“找到一篇文章”走到“某条结论有可定位的来源”。本人推进检索、阅读和结构化证据链路。

输入是研究查询及所需证据条件；`sources` 保存来源事实，`evidence` 保存证据条目，`claims` 通过 evidence_id 引用证据。PMID、DOI、URL 或数据库记录 ID 用于归一来源身份。

1. 检索子 Agent 迭代调用检索/阅读能力，收集提供商返回的数据。
2. EvidencePack 归一化结构并去重；输出预算被截断时记录缺失数量与限制，完整数据通过资源或执行记录查阅。
3. reconciliation 使用 Provider 的来源事实校正模型生成的定位元数据。
4. sufficiency 检查结论是否引用已返回证据、是否满足来源数量等约束，并排除明确失败或 fallback 结果被误判完成。

可先阅读测试中 `test_reconciliation_replaces_model_locator_metadata_with_authoritative_values`、`test_evidence_pack_sufficiency_requires_every_claim_to_reference_returned_evidence` 和 `test_web_fallback_does_not_claim_query_is_satisfied`。

边界：结构化证据校验不等于科学结论正确；检索无结果、证据不足和工具错误需要区别记录。本轮没有访问文献服务，也没有重新开展真实科研验证。

## 代码阅读顺序

1. [build_retrieval_subagent_tool](../source/agent/src/subagents/retrieval_subagent.py#L3238)
2. [_reconcile_evidence_pack_sources](../source/agent/src/subagents/retrieval_subagent.py#L2854)
3. [_evidence_pack_is_sufficient](../source/agent/src/subagents/retrieval_subagent.py#L2968)
4. [normalize_evidence_pack](../source/agent/src/subagents/evidence_pack.py#L177)
5. [materialize_source_records](../source/agent/src/subagents/source_evidence_contracts.py#L189)
6. [document_read_service.py](../source/server/backend/services/document_read_service.py)
7. [test_retrieval_subagent_submit_tool.py](../source/agent/tests/test_retrieval_subagent_submit_tool.py)
8. [test_source_evidence_contracts.py](../source/agent/tests/test_source_evidence_contracts.py)

[返回总览](../README.md) · [来源与整理边界](PROVENANCE.md) · [依赖与验证](DEPENDENCIES.md)
