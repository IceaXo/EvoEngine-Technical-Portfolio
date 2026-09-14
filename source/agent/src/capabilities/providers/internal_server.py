"""Context-aware provider for hidden Server-owned atomic capabilities."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from src.capabilities.models import (
    ArtifactRef,
    CapabilityError,
    CapabilityErrorKind,
    CapabilityNeedsInput,
    CapabilitySpec,
    CapabilityStatus,
    DiscoveryMatchMode,
    DiscoveryPolicy,
    DiscoveryVisibility,
    Permission,
    ProviderType,
    RegistrationStatus,
    ResourceRef,
)
from src.capabilities.providers.base import ProviderContext, ProviderInvocationResult
from src.schemas.hitl import HumanQuestionBundle, HumanQuestionItem, HumanQuestionOption
from src.services.deliverable_capability_client import (
    DeliverableCapabilityClient,
    DeliverableCapabilityClientError,
)
from src.services.image_region_execution import (
    ImageRegionExecutionError,
    execute_image_region_edit,
)
from src.services.office_delivery import execute_office_render, freeze_render_arguments


InternalHandler = Callable[
    [dict[str, Any], list[dict[str, Any]], ProviderContext],
    Awaitable[ProviderInvocationResult],
]


@dataclass(frozen=True)
class _Binding:
    handler: InternalHandler


class InternalServerProvider:
    provider_type = ProviderType.INTERNAL_SERVER

    def __init__(self) -> None:
        self._bindings: dict[str, _Binding] = {}

    def register(self, spec: CapabilitySpec, handler: InternalHandler) -> None:
        if spec.provider_type != ProviderType.INTERNAL_SERVER:
            raise ValueError("InternalServerProvider only accepts internal_server specs")
        if spec.capability_id in self._bindings:
            raise ValueError(f"internal capability already bound: {spec.capability_id}")
        self._bindings[spec.capability_id] = _Binding(handler=handler)

    async def invoke(
        self,
        *,
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        binding = self._bindings.get(context.capability_id)
        if binding is None:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Internal capability is unavailable",
                error=CapabilityError(
                    kind=CapabilityErrorKind.PROVIDER_UNAVAILABLE,
                    message="internal Server capability binding is unavailable",
                ),
            )
        return await binding.handler(arguments, input_refs, context)


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _authorization_result(
    *,
    arguments: dict[str, Any],
    context: ProviderContext,
    action: str,
    title: str,
) -> ProviderInvocationResult:
    target = ResourceRef.model_validate(arguments.get("target"))
    call_id = str(context.metadata.get("call_id") or "")
    request_digest = _canonical_digest(arguments)
    question_id = f"authorize:{call_id}"
    approve_id = "approve"
    reject_id = "reject"
    bundle = HumanQuestionBundle(
        bundle_id=f"capability:{call_id}",
        bundle_title=title,
        bundle_summary="确认后只执行这一次已冻结的候选操作，不会接受或删除文件。",
        questions=[
            HumanQuestionItem(
                question_id=question_id,
                label=title,
                description="目标、版本和内容摘要已冻结；目标变化后授权自动失效。",
                kind="single",
                required=True,
                options=[
                    HumanQuestionOption(option_id=approve_id, label="确认执行"),
                    HumanQuestionOption(option_id=reject_id, label="取消"),
                ],
                allow_other=False,
                max_select=1,
            )
        ],
    )
    challenge = _canonical_digest(
        {
            "call_id": call_id,
            "capability_id": context.capability_id,
            "capability_version": context.capability_version,
            "registry_snapshot_id": context.metadata.get("registry_snapshot_id") or "",
            "action": action,
            "target": target.model_dump(mode="json"),
            "request_digest": request_digest,
        }
    )
    return ProviderInvocationResult(
        status=CapabilityStatus.NEEDS_INPUT,
        summary="This operation requires user confirmation",
        error=CapabilityError(
            kind=CapabilityErrorKind.NEEDS_INPUT,
            message="user confirmation is required",
        ),
        needs_input_control=CapabilityNeedsInput(
            question_bundle=bundle,
            pending_call_id=call_id,
            capability_id=context.capability_id,
            capability_version=context.capability_version,
            registry_snapshot_id=str(context.metadata.get("registry_snapshot_id") or ""),
            action=action,
            target=target,
            target_version_id=target.conversation_file_id,
            target_lock_version=int(
                arguments.get(
                    "expected_lock_version",
                    arguments.get("expected_source_lock_version", 0),
                )
            ),
            target_sha256=target.sha256,
            request_digest=request_digest,
            authorization_challenge=challenge,
            authorization_question_id=question_id,
            approve_option_id=approve_id,
            reject_option_id=reject_id,
        ),
    )


def _authorization_transport(context: ProviderContext, request_digest: str) -> dict[str, Any] | None:
    private = context.metadata.get("capability_authorization")
    if not isinstance(private, dict) or not str(private.get("grant_id") or ""):
        return None
    authorized_digest = str(
        private.get("authorized_request_digest")
        or private.get("request_digest")
        or ""
    )
    effect_capability_id = str(private.get("effect_capability_id") or "")
    effect_request_digest = str(private.get("effect_request_digest") or "")
    if bool(effect_capability_id) != bool(effect_request_digest):
        raise ValueError("capability authorization effect binding is incomplete")
    if effect_request_digest:
        if (
            effect_capability_id != context.capability_id
            or effect_request_digest != request_digest
        ):
            raise ValueError("capability authorization effect digest mismatch")
    elif authorized_digest != request_digest:
        raise ValueError("capability authorization digest mismatch")
    result = {
        "grant_id": str(private["grant_id"]),
        "parent_request_id": str(private.get("parent_request_id") or ""),
        "continuation_request_id": str(private.get("continuation_request_id") or ""),
        "pending_call_id": str(private.get("pending_call_id") or ""),
        "request_digest": authorized_digest,
        "capability_id": str(
            private.get("authorized_capability_id") or context.capability_id
        ),
    }
    if effect_request_digest:
        result.update(
            effect_capability_id=effect_capability_id,
            effect_request_digest=effect_request_digest,
        )
    return result


def _text_candidate_handler(client: DeliverableCapabilityClient) -> InternalHandler:
    async def _handler(
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        if input_refs:
            raise ValueError("text candidate capability does not accept input_refs")
        request_digest = _canonical_digest(arguments)
        authorization = _authorization_transport(context, request_digest)
        if authorization is None:
            return _authorization_result(
                arguments=arguments,
                context=context,
                action="create_text_candidate",
                title="创建一个新的文本候选版本？",
            )
        try:
            payload = await asyncio.to_thread(
                client.invoke,
                capability_id=context.capability_id,
                arguments=arguments,
                authorization=authorization,
                scope={
                    "user_id": context.user_id,
                    "project_id": context.project_id,
                    "conversation_id": context.conversation_id,
                },
            )
            candidate = dict(payload.get("candidate") or {})
            file_id = int(candidate["candidate_file_id"])
            root_file_id = int(candidate.get("root_file_id") or 0)
            if root_file_id <= 0:
                raise ValueError("text candidate response omitted the visible root file")
            resource = ResourceRef(
                resource_id=f"resource:conversation-file:{file_id}",
                uri=f"conversation-file://{file_id}",
                kind="conversation_file",
                storage="conversation_file",
                media_type=str(candidate.get("mime_type") or "text/plain"),
                size_bytes=int(candidate.get("size_bytes") or 0),
                sha256=str(candidate["content_sha256"]),
                conversation_file_id=file_id,
                metadata={"role": "candidate", "accepted": False},
            )
            artifact = ArtifactRef(
                artifact_id=f"conversation-file:{root_file_id}",
                artifact_key="text_candidate",
                conversation_file_id=root_file_id,
                file_name=str(candidate.get("file_name") or "candidate.txt"),
                mime_type=resource.media_type,
                size_bytes=resource.size_bytes,
                sha256=resource.sha256,
                source_type="agent_candidate",
                registration_status=RegistrationStatus.REGISTERED,
            )
            return ProviderInvocationResult(
                status=CapabilityStatus.SUCCEEDED,
                summary="Text candidate created; it is not selected or accepted",
                data={
                    "candidate_file_id": file_id,
                    "root_file_id": candidate.get("root_file_id"),
                    "lineage_id": candidate.get("lineage_id"),
                    "version_id": candidate.get("version_id"),
                    "active_version_file_id": candidate.get("active_version_file_id"),
                    "accepted_version_file_id": candidate.get("accepted_version_file_id"),
                    "lock_version": candidate.get("lock_version"),
                },
                resources=[resource],
                artifacts=[artifact],
                idempotency_reused=bool(payload.get("idempotent_replay")),
            )
        except DeliverableCapabilityClientError as exc:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Text candidate creation failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.EXECUTION_ERROR,
                    message=f"text candidate failed [{exc.code}]",
                ),
            )

    return _handler


def _image_region_candidate_handler(
    client: DeliverableCapabilityClient,
) -> InternalHandler:
    async def _handler(
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        if input_refs:
            raise ValueError("image region candidate capability does not accept input_refs")
        request_digest = _canonical_digest(arguments)
        authorization = _authorization_transport(context, request_digest)
        if authorization is None:
            return _authorization_result(
                arguments=arguments,
                context=context,
                action="create_image_region_candidate",
                title="对所选图片区域执行修改并创建候选版本？",
            )
        try:
            prepared = await asyncio.to_thread(
                client.invoke,
                capability_id=context.capability_id,
                arguments=arguments,
                authorization=authorization,
                scope={
                    "user_id": context.user_id,
                    "project_id": context.project_id,
                    "conversation_id": context.conversation_id,
                },
            )
            payload = await asyncio.to_thread(
                execute_image_region_edit,
                prepared=prepared,
                client=client,
            )
            candidate = dict(payload.get("candidate") or {})
            resource = ResourceRef.model_validate(payload.get("candidate_resource"))
            file_id = int(candidate.get("candidate_file_id") or resource.conversation_file_id or 0)
            root_file_id = int(candidate.get("root_file_id") or 0)
            if root_file_id <= 0:
                raise ImageRegionExecutionError("image candidate response omitted the visible root file")
            artifact = ArtifactRef(
                artifact_id=f"conversation-file:{root_file_id}",
                artifact_key="image_region_candidate",
                conversation_file_id=root_file_id,
                file_name=str(candidate.get("file_name") or "image_region_candidate.png"),
                mime_type=resource.media_type,
                size_bytes=resource.size_bytes,
                sha256=resource.sha256,
                source_type="image_region_candidate",
                registration_status=RegistrationStatus.REGISTERED,
            )
            return ProviderInvocationResult(
                status=CapabilityStatus.SUCCEEDED,
                summary="Image region candidate created; it is not selected or accepted",
                data={
                    "candidate_file_id": file_id,
                    "root_file_id": root_file_id,
                    "lineage_id": candidate.get("lineage_id"),
                    "version_id": candidate.get("version_id"),
                    "active_version_file_id": candidate.get("active_version_file_id"),
                    "accepted_version_file_id": candidate.get("accepted_version_file_id"),
                    "lock_version": candidate.get("lock_version"),
                    "artifact_manifest_resource": payload.get("artifact_manifest_resource"),
                    "runner": payload.get("runner"),
                },
                resources=[resource],
                artifacts=[artifact],
                idempotency_reused=bool(
                    prepared.get("idempotent_replay") or payload.get("idempotent_replay")
                ),
            )
        except (DeliverableCapabilityClientError, ImageRegionExecutionError) as exc:
            code = exc.code if isinstance(exc, DeliverableCapabilityClientError) else "IMAGE_REGION_EXECUTION_FAILED"
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Image region candidate creation failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.EXECUTION_ERROR,
                    message=f"image region candidate failed [{code}]",
                ),
            )

    return _handler


def _office_blueprint_target(arguments: dict[str, Any]) -> ResourceRef:
    digest = _canonical_digest(arguments)
    blueprint = str(arguments.get("blueprint_markdown") or "")
    identity = f"office-blueprint/{digest}"
    return ResourceRef(
        resource_id=f"resource:conversation-turn:{identity}",
        uri=f"conversation-turn://{identity}",
        kind="conversation_turn",
        storage="conversation_turn",
        media_type="text/markdown; charset=utf-8",
        size_bytes=len(blueprint.encode("utf-8")),
        sha256=hashlib.sha256(blueprint.encode("utf-8")).hexdigest(),
        metadata={
            "role": "office_blueprint_contract",
            "contract_sha256": digest,
        },
    )


def _office_blueprint_create_handler(
    client: DeliverableCapabilityClient,
) -> InternalHandler:
    async def _handler(
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        if input_refs:
            raise ValueError("Office blueprint capability does not accept input_refs")
        frozen = {
            **arguments,
            "target": _office_blueprint_target(arguments).model_dump(mode="json"),
        }
        authorization = _authorization_transport(context, _canonical_digest(frozen))
        if authorization is None:
            return _authorization_result(
                arguments=frozen,
                context=context,
                action="create_office_blueprint",
                title="创建 Office 内容蓝图并进入版本验收？",
            )
        try:
            payload = await asyncio.to_thread(
                client.invoke,
                capability_id=context.capability_id,
                arguments=arguments,
                authorization=authorization,
                scope={
                    "user_id": context.user_id,
                    "project_id": context.project_id,
                    "conversation_id": context.conversation_id,
                },
            )
            blueprint = dict(payload.get("blueprint") or {})
            file_id = int(blueprint.get("blueprint_file_id") or 0)
            resource = ResourceRef(
                resource_id=f"resource:conversation-file:{file_id}",
                uri=f"conversation-file://{file_id}",
                kind="conversation_file",
                storage="conversation_file",
                media_type=str(blueprint.get("mime_type") or "text/markdown"),
                size_bytes=int(blueprint.get("size_bytes") or 0),
                sha256=str(blueprint.get("content_sha256") or ""),
                conversation_file_id=file_id,
                metadata={"role": "office_blueprint", "accepted": False},
            )
            artifact = ArtifactRef(
                artifact_id=f"conversation-file:{file_id}",
                artifact_key="office_blueprint",
                conversation_file_id=file_id,
                file_name=str(blueprint.get("file_name") or "Office-内容蓝图.md"),
                mime_type=resource.media_type,
                size_bytes=resource.size_bytes,
                sha256=resource.sha256,
                source_type="agent_generated",
                registration_status=RegistrationStatus.REGISTERED,
            )
            return ProviderInvocationResult(
                status=CapabilityStatus.SUCCEEDED,
                summary="Office blueprint created; explicit version acceptance is still required",
                data={"blueprint": blueprint},
                resources=[resource],
                artifacts=[artifact],
                idempotency_reused=bool(payload.get("idempotent_replay")),
            )
        except DeliverableCapabilityClientError as exc:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Office blueprint creation failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.EXECUTION_ERROR,
                    message=f"Office blueprint failed [{exc.code}]",
                ),
            )

    return _handler


def _office_blueprint_patch_handler(
    client: DeliverableCapabilityClient,
) -> InternalHandler:
    async def _handler(
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        if input_refs:
            raise ValueError("Office blueprint patch does not accept input_refs")
        request_digest = _canonical_digest(arguments)
        authorization = _authorization_transport(context, request_digest)
        if authorization is None:
            return _authorization_result(
                arguments=arguments,
                context=context,
                action="patch_office_blueprint",
                title="按所选 Office 内容修订蓝图并创建候选版本？",
            )
        try:
            payload = await asyncio.to_thread(
                client.invoke,
                capability_id=context.capability_id,
                arguments=arguments,
                authorization=authorization,
                scope={
                    "user_id": context.user_id,
                    "project_id": context.project_id,
                    "conversation_id": context.conversation_id,
                },
            )
            candidate = dict(payload.get("candidate") or {})
            file_id = int(candidate.get("candidate_file_id") or 0)
            root_file_id = int(candidate.get("root_file_id") or 0)
            if file_id <= 0 or root_file_id <= 0:
                raise ValueError("Office blueprint candidate response omitted file identity")
            resource = ResourceRef(
                resource_id=f"resource:conversation-file:{file_id}",
                uri=f"conversation-file://{file_id}",
                kind="conversation_file",
                storage="conversation_file",
                media_type=str(candidate.get("mime_type") or "text/markdown"),
                size_bytes=int(candidate.get("size_bytes") or 0),
                sha256=str(candidate.get("content_sha256") or ""),
                conversation_file_id=file_id,
                metadata={"role": "office_blueprint_candidate", "accepted": False},
            )
            return ProviderInvocationResult(
                status=CapabilityStatus.SUCCEEDED,
                summary="Office blueprint candidate created from a canonical selection",
                data={"candidate": candidate},
                resources=[resource],
                artifacts=[
                    ArtifactRef(
                        artifact_id=f"conversation-file:{root_file_id}",
                        artifact_key="office_blueprint_candidate",
                        conversation_file_id=root_file_id,
                        file_name=str(candidate.get("file_name") or "Office-内容蓝图.md"),
                        mime_type=resource.media_type,
                        size_bytes=resource.size_bytes,
                        sha256=resource.sha256,
                        source_type="agent_candidate",
                        registration_status=RegistrationStatus.REGISTERED,
                    )
                ],
                idempotency_reused=bool(payload.get("idempotent_replay")),
            )
        except DeliverableCapabilityClientError as exc:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Office blueprint patch failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.EXECUTION_ERROR,
                    message=f"Office blueprint patch failed [{exc.code}]",
                ),
            )

    return _handler


def _office_render_handler(
    client: DeliverableCapabilityClient,
) -> InternalHandler:
    async def _handler(
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        if input_refs:
            raise ValueError("Office render capability does not accept input_refs")
        frozen = freeze_render_arguments(arguments)
        authorization = _authorization_transport(context, _canonical_digest(frozen))
        if authorization is None:
            return _authorization_result(
                arguments=frozen,
                context=context,
                action="render_office",
                title="按已确认蓝图生成排版源码和 Office 成品？",
            )
        try:
            payload = await asyncio.to_thread(
                execute_office_render,
                client=client,
                arguments=frozen,
                authorization=authorization,
                user_id=context.user_id,
                project_id=context.project_id,
                conversation_id=context.conversation_id,
                request_id=context.request_id,
            )
            completion = dict(payload.get("completion") or {})
            delivery = dict(completion.get("delivery") or {})
            result_file = dict(delivery.get("result_file") or {})
            run = dict(payload.get("run") or {})
            file_id = int(
                result_file.get("conversation_file_id")
                or delivery.get("root_file_id")
                or 0
            )
            artifacts_by_path = {
                str(item.get("path") or ""): item
                for item in run.get("artifacts") or []
                if isinstance(item, dict)
            }
            primary = dict(artifacts_by_path.get(str(run.get("output_path") or "")) or {})
            resource = ResourceRef(
                resource_id=f"resource:conversation-file:{file_id}",
                uri=f"conversation-file://{file_id}",
                kind="conversation_file",
                storage="conversation_file",
                media_type=str(result_file.get("mime_type") or "application/octet-stream"),
                size_bytes=int(result_file.get("size_bytes") or 0),
                sha256=str(result_file.get("sha256") or ""),
                conversation_file_id=file_id,
                metadata={
                    "role": "office_output",
                    "candidate": delivery.get("candidate_file_id") is not None,
                    "candidate_file_id": delivery.get("candidate_file_id"),
                    "rendered_file_id": delivery.get("rendered_file_id"),
                },
            )
            return ProviderInvocationResult(
                status=CapabilityStatus.SUCCEEDED,
                summary="Office source and deliverable rendered through the unified artifact runner",
                data={"delivery": delivery, "run_id": run.get("run_id")},
                resources=[resource],
                artifacts=[
                    ArtifactRef(
                        artifact_id=f"conversation-file:{file_id}",
                        artifact_key="office_output",
                        conversation_file_id=file_id,
                        file_name=str(result_file.get("file_name") or "office-output"),
                        mime_type=resource.media_type,
                        size_bytes=resource.size_bytes,
                        sha256=resource.sha256,
                        source_type="artifact_generated",
                        registration_status=RegistrationStatus.REGISTERED,
                    )
                ],
                idempotency_reused=bool(completion.get("idempotent_replay")),
            )
        except (DeliverableCapabilityClientError, RuntimeError, ValueError) as exc:
            code = exc.code if isinstance(exc, DeliverableCapabilityClientError) else "OFFICE_RENDER_FAILED"
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Office render failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.EXECUTION_ERROR,
                    message=f"Office render failed [{code}]",
                ),
            )

    return _handler


def _server_passthrough_handler(
    client: DeliverableCapabilityClient,
    *,
    action: str | None,
    title: str = "",
) -> InternalHandler:
    async def _handler(
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        if input_refs:
            raise ValueError("scientific figure atom does not accept input_refs")
        request_digest = _canonical_digest(arguments)
        authorization = None
        if action:
            authorization = _authorization_transport(context, request_digest)
            if authorization is None:
                return _authorization_result(
                    arguments=arguments,
                    context=context,
                    action=action,
                    title=title,
                )
        try:
            payload = await asyncio.to_thread(
                client.invoke,
                capability_id=context.capability_id,
                arguments=arguments,
                authorization=authorization,
                scope={
                    "user_id": context.user_id,
                    "project_id": context.project_id,
                    "conversation_id": context.conversation_id,
                },
            )
            status = str((payload.get("run") or {}).get("status") or payload.get("status") or "")
            return ProviderInvocationResult(
                status=(
                    CapabilityStatus.PENDING
                    if status in {"prepared", "running"}
                    else CapabilityStatus.SUCCEEDED
                ),
                summary="Scientific figure run submitted" if action else "Scientific figure status read",
                data=payload,
            )
        except DeliverableCapabilityClientError as exc:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Scientific figure capability failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.EXECUTION_ERROR,
                    message=f"scientific figure failed [{exc.code}]",
                ),
            )

    return _handler


def _scientific_context_handler(
    client: DeliverableCapabilityClient,
) -> InternalHandler:
    async def _handler(
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        if input_refs:
            raise ValueError("scientific figure context does not accept input_refs")
        try:
            payload = await asyncio.to_thread(
                client.invoke,
                capability_id=context.capability_id,
                arguments=arguments,
                authorization=None,
                scope={
                    "user_id": context.user_id,
                    "project_id": context.project_id,
                    "conversation_id": context.conversation_id,
                },
            )
            context_payload = dict(payload.get("context") or {})
            source = ResourceRef.model_validate(context_payload.get("source_resource"))
            context_payload["source_resource"] = source.model_dump(mode="json")
            return ProviderInvocationResult(
                status=CapabilityStatus.SUCCEEDED,
                summary="Managed scientific figure context read",
                data={"scientific_context": context_payload},
                resources=[source],
            )
        except DeliverableCapabilityClientError as exc:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Scientific figure context read failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.EXECUTION_ERROR,
                    message=f"scientific figure context failed [{exc.code}]",
                ),
            )

    return _handler


def internal_deliverable_capability_specs() -> list[CapabilitySpec]:
    hidden = DiscoveryPolicy(
        visibility=DiscoveryVisibility.HIDDEN,
        match_mode=DiscoveryMatchMode.HIDDEN,
    )
    resource_schema = ResourceRef.model_json_schema()
    edit_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["operation", "start_line", "expected_text"],
        "properties": {
            "operation": {"enum": ["replace", "delete", "insert_before", "insert_after"]},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": ["integer", "null"], "minimum": 1},
            "expected_text": {"type": "string", "maxLength": 500000},
            "replacement_text": {"type": "string", "maxLength": 500000},
        },
    }
    image_edit_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["operation"],
        "properties": {
            "operation": {"enum": ["blur", "pixelate", "grayscale", "fill", "crop"]},
            "blur_radius": {"type": "integer", "minimum": 1, "maximum": 50, "default": 12},
            "pixel_size": {"type": "integer", "minimum": 2, "maximum": 64, "default": 12},
            "fill_color": {
                "type": "string",
                "pattern": "^#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?$",
                "default": "#000000",
            },
        },
    }
    office_block_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["type"],
        "properties": {
            "type": {"enum": ["heading", "paragraph", "bullets", "list", "table", "image", "figure", "page_break"]},
            "text": {"type": "string", "maxLength": 100000},
            "level": {"type": "integer", "minimum": 1, "maximum": 6},
            "items": {"type": "array", "maxItems": 500, "items": {"type": "string", "maxLength": 10000}},
            "headers": {"type": "array", "maxItems": 100, "items": {"type": "string", "maxLength": 10000}},
            "rows": {"type": "array", "maxItems": 1000, "items": {"type": "array", "maxItems": 100, "items": {"type": "string", "maxLength": 10000}}},
            "binding_key": {"type": "string", "maxLength": 120},
            "asset_binding": {"type": "string", "maxLength": 120},
            "caption": {"type": "string", "maxLength": 4000},
        },
    }
    office_slide_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "layout": {"enum": ["title", "bullets", "figure"]},
            "title": {"type": "string", "maxLength": 10000},
            "subtitle": {"type": "string", "maxLength": 10000},
            "body": {"type": "string", "maxLength": 100000},
            "bullets": {"type": "array", "maxItems": 100, "items": {"type": "string", "maxLength": 10000}},
            "image_key": {"type": "string", "maxLength": 120},
            "caption": {"type": "string", "maxLength": 4000},
        },
    }
    office_source_spec_schema = {
        "type": "object",
        "additionalProperties": False,
        "maxProperties": 4,
        "properties": {
            "kind": {"enum": ["docx", "pptx"]},
            "title": {"type": "string", "maxLength": 10000},
            "blocks": {"type": "array", "maxItems": 2000, "items": office_block_schema},
            "slides": {"type": "array", "maxItems": 60, "items": office_slide_schema},
        },
    }
    office_asset_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["binding_key", "conversation_file_id"],
        "properties": {
            "binding_key": {"type": "string", "pattern": "^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$"},
            "conversation_file_id": {"type": "integer", "minimum": 1},
            "sha256": {"type": "string", "pattern": "^(?:|[a-f0-9]{64})$"},
            "caption": {"type": "string", "maxLength": 4000},
            "alt_text": {"type": "string", "maxLength": 4000},
        },
    }
    common = {
        "provider_type": ProviderType.INTERNAL_SERVER,
        "provider_ref": {"handler_kind": "deliverable_atom"},
        "input_refs_schema": {"type": "array", "maxItems": 0},
        "permission": Permission.WRITE,
        "discovery": hidden,
        "requires_confirmation": True,
    }
    return [
        CapabilitySpec(
            capability_id="office.blueprint.create",
            version="1.0.0",
            display_name="Create Office content blueprint",
            description="Create a versioned DOCX/PPTX content blueprint; it remains unaccepted.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["deliverable_kind", "name", "blueprint_markdown", "idempotency_key"],
                "properties": {
                    "deliverable_kind": {"enum": ["docx", "pptx"]},
                    "name": {"type": "string", "minLength": 1, "maxLength": 180},
                    "blueprint_markdown": {"type": "string", "minLength": 1, "maxLength": 250000},
                    "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 128},
                },
            },
            output_schema={"type": "object"},
            side_effects=["creates_office_blueprint"],
            **common,
        ),
        CapabilitySpec(
            capability_id="office.blueprint.patch",
            version="1.0.0",
            display_name="Patch Office content blueprint",
            description="Create an unaccepted blueprint candidate from a canonical Office text/figure selection.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["target", "selection", "expected_lock_version", "idempotency_key", "edits"],
                "properties": {
                    "target": resource_schema,
                    "selection": resource_schema,
                    "expected_lock_version": {"type": "integer", "minimum": 0},
                    "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 128},
                    "summary": {"type": ["string", "null"], "maxLength": 1000},
                    "edits": {"type": "array", "minItems": 1, "maxItems": 100, "items": edit_schema},
                },
            },
            output_schema={"type": "object"},
            side_effects=["creates_office_blueprint_candidate"],
            **common,
        ),
        CapabilitySpec(
            capability_id="office.render.execute",
            version="1.0.0",
            display_name="Render accepted Office blueprint",
            description="Render DOCX/PPTX plus layout source and QA through the unified artifact runner.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["target", "blueprint_root_file_id", "expected_blueprint_lock_version", "deliverable_kind", "name", "source_spec", "assets", "idempotency_key"],
                "properties": {
                    "target": resource_schema,
                    "blueprint_root_file_id": {"type": "integer", "minimum": 1},
                    "expected_blueprint_lock_version": {"type": "integer", "minimum": 0},
                    "deliverable_kind": {"enum": ["docx", "pptx"]},
                    "name": {"type": "string", "minLength": 1, "maxLength": 180},
                    "source_spec": office_source_spec_schema,
                    "assets": {"type": "array", "maxItems": 100, "items": office_asset_schema},
                    "output_file_name": {"type": ["string", "null"], "maxLength": 180},
                    "office_root_file_id": {"type": ["integer", "null"], "minimum": 1},
                    "expected_office_version_file_id": {"type": ["integer", "null"], "minimum": 1},
                    "expected_office_lock_version": {"type": ["integer", "null"], "minimum": 0},
                    "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 128},
                },
            },
            output_schema={"type": "object"},
            side_effects=["renders_office_with_unified_runner", "attaches_office_version"],
            **common,
        ),
        CapabilitySpec(
            capability_id="conversation_file.text_candidate.create",
            version="1.0.0",
            display_name="Create text candidate",
            description="Create a hidden Markdown/text candidate without selecting or accepting it.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["target", "content_text", "expected_lock_version", "idempotency_key"],
                "properties": {
                    "target": resource_schema,
                    "content_text": {"type": "string", "maxLength": 500000},
                    "summary": {"type": ["string", "null"], "maxLength": 1000},
                    "expected_lock_version": {"type": "integer", "minimum": 0},
                    "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 128},
                },
            },
            output_schema={"type": "object"},
            side_effects=["creates_hidden_candidate"],
            **common,
        ),
        CapabilitySpec(
            capability_id="conversation_file.image_region_candidate.create",
            version="1.0.0",
            display_name="Create image region candidate",
            description=(
                "Apply one fixed guarded operation to a Server-verified ordinary-image region, "
                "then attach a hidden candidate without selecting or accepting it."
            ),
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "target", "selection", "edit", "expected_lock_version", "idempotency_key",
                ],
                "properties": {
                    "target": resource_schema,
                    "selection": resource_schema,
                    "edit": image_edit_schema,
                    "summary": {"type": ["string", "null"], "maxLength": 1000},
                    "expected_lock_version": {"type": "integer", "minimum": 0},
                    "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 128},
                },
            },
            output_schema={"type": "object"},
            side_effects=["runs_guarded_image_edit", "creates_hidden_candidate"],
            **common,
        ),
        CapabilitySpec(
            capability_id="scientific_figure.context.read",
            version="1.0.0",
            display_name="Read scientific figure context",
            description="Resolve one managed figure to its active source ResourceRef and exact lock state.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["conversation_file_id"],
                "properties": {
                    "conversation_file_id": {"type": "integer", "minimum": 1}
                },
            },
            output_schema={"type": "object"},
            side_effects=[],
            permission=Permission.READ,
            requires_confirmation=False,
            provider_type=ProviderType.INTERNAL_SERVER,
            provider_ref={"handler_kind": "deliverable_atom"},
            input_refs_schema={"type": "array", "maxItems": 0},
            discovery=hidden,
        ),
        CapabilitySpec(
            capability_id="scientific_figure.run.submit",
            version="1.0.0",
            display_name="Submit scientific figure run",
            description="Create one WP-04 scientific figure source candidate and run.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "target", "family_id", "expected_family_lock_version",
                    "expected_source_lock_version", "expected_source_version_file_id",
                    "idempotency_key", "edits",
                ],
                "properties": {
                    "target": resource_schema,
                    "family_id": {"type": "integer", "minimum": 1},
                    "expected_family_lock_version": {"type": "integer", "minimum": 0},
                    "expected_source_lock_version": {"type": "integer", "minimum": 0},
                    "expected_source_version_file_id": {"type": "integer", "minimum": 1},
                    "idempotency_key": {"type": "string", "minLength": 1, "maxLength": 128},
                    "edits": {"type": "array", "minItems": 1, "maxItems": 100, "items": edit_schema},
                },
            },
            output_schema={"type": "object"},
            side_effects=["creates_scientific_figure_run"],
            **common,
        ),
        CapabilitySpec(
            capability_id="scientific_figure.run.status",
            version="1.0.0",
            display_name="Read scientific figure run status",
            description="Read or reconcile one existing WP-04 run without creating a new run.",
            input_schema={
                "type": "object",
                "additionalProperties": False,
                "required": ["family_id", "run_id"],
                "properties": {
                    "family_id": {"type": "integer", "minimum": 1},
                    "run_id": {"type": "string", "pattern": "^sfr_[a-f0-9]{32}$"},
                },
            },
            output_schema={"type": "object"},
            side_effects=[],
            permission=Permission.READ,
            requires_confirmation=False,
            provider_type=ProviderType.INTERNAL_SERVER,
            provider_ref={"handler_kind": "deliverable_atom"},
            input_refs_schema={"type": "array", "maxItems": 0},
            discovery=hidden,
        ),
    ]


def register_internal_deliverable_capabilities(
    *,
    registry: Any,
    provider: InternalServerProvider,
    client: DeliverableCapabilityClient | None = None,
) -> tuple[str, ...]:
    bound_client = client or DeliverableCapabilityClient()
    handlers: dict[str, InternalHandler] = {
        "office.blueprint.create": _office_blueprint_create_handler(bound_client),
        "office.blueprint.patch": _office_blueprint_patch_handler(bound_client),
        "office.render.execute": _office_render_handler(bound_client),
        "conversation_file.text_candidate.create": _text_candidate_handler(bound_client),
        "conversation_file.image_region_candidate.create": _image_region_candidate_handler(bound_client),
        "scientific_figure.context.read": _scientific_context_handler(bound_client),
        "scientific_figure.run.submit": _server_passthrough_handler(
            bound_client,
            action="submit_scientific_figure_run",
            title="创建科研图源码候选并启动重跑？",
        ),
        "scientific_figure.run.status": _server_passthrough_handler(
            bound_client,
            action=None,
        ),
    }
    registered: list[str] = []
    for spec in internal_deliverable_capability_specs():
        registry.register_spec(spec)
        provider.register(spec, handlers[spec.capability_id])
        registered.append(f"{spec.capability_id}@{spec.version}")
    return tuple(registered)
