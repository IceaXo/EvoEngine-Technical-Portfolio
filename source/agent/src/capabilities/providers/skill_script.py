"""Structured subprocess provider for executable Skill contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from src.capabilities.models import (
    ArtifactRef,
    CapabilityError,
    CapabilityErrorKind,
    CapabilityOutcome,
    CapabilityStatus,
    DrawerSection,
    EvidenceRef,
    ProviderType,
    RegistrationStatus,
    ResourceRef,
)
from src.capabilities.providers.base import (
    ProviderArtifactSource,
    ProviderContext,
    ProviderInvocationResult,
)
from src.capabilities.source_candidates import build_source_candidate_ref


logger = logging.getLogger("evoengine.capability.skill_script")


def _capability_outcome_from_payload(payload: dict[str, Any]) -> CapabilityOutcome:
    """Read an explicit result semantic contract without inferring legacy data."""
    raw_outcome = payload.get("capability_outcome")
    if raw_outcome is None:
        nested = payload.get("result")
        if isinstance(nested, dict):
            raw_outcome = nested.get("capability_outcome")
    if raw_outcome is None:
        return CapabilityOutcome()
    return CapabilityOutcome.model_validate(raw_outcome)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_output_dir(output_dir: Path) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative_path = path.resolve().relative_to(output_dir.resolve()).as_posix()
            stat = path.stat()
        except (OSError, ValueError):
            continue
        snapshot[relative_path] = (int(stat.st_size), int(stat.st_mtime_ns))
    return snapshot


def _safe_script_path(provider_ref: dict[str, Any]) -> tuple[Path, Path]:
    skill_dir = Path(str(provider_ref.get("skill_dir") or "")).resolve()
    scripts_root = (skill_dir / "scripts").resolve()
    script_value = str(provider_ref.get("script_path") or "").strip()
    script_path = (
        Path(script_value).resolve()
        if script_value
        else (scripts_root / str(provider_ref.get("script_name") or "")).resolve()
    )
    try:
        script_path.relative_to(scripts_root)
    except ValueError as exc:
        raise ValueError("Skill script path escapes scripts directory") from exc
    if not script_path.is_file():
        raise FileNotFoundError("Skill script is unavailable")
    return skill_dir, script_path


def _artifact_refs(
    output_dir: Path,
    context: ProviderContext,
    before_snapshot: dict[str, tuple[int, int]],
) -> tuple[list[ArtifactRef], list[ProviderArtifactSource], list[str]]:
    files: list[Path] = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            relative_path = path.resolve().relative_to(output_dir.resolve()).as_posix()
            stat = path.stat()
        except (OSError, ValueError):
            continue
        if before_snapshot.get(relative_path) == (int(stat.st_size), int(stat.st_mtime_ns)):
            continue
        files.append(path)
    artifacts: list[ArtifactRef] = []
    sources: list[ProviderArtifactSource] = []
    matched_keys: set[str] = set()
    for path in files:
        extension = path.suffix.lower().lstrip(".")
        artifact_spec = next(
            (
                item
                for item in context.artifact_specs
                if item.file_name == path.name
                and (not item.file_types or extension in item.file_types)
            ),
            None,
        )
        if artifact_spec is None:
            artifact_spec = next(
                (
                    item
                    for item in context.artifact_specs
                    if item.file_name is None
                    and (not item.file_types or extension in item.file_types)
                ),
                None,
            )
        drawer_section = artifact_spec.drawer_section if artifact_spec else DrawerSection.TEMPORARY_OUTPUT
        artifact_key = artifact_spec.artifact_key if artifact_spec else ""
        if artifact_key:
            matched_keys.add(artifact_key)
        digest = _sha256(path)
        artifact = ArtifactRef(
            artifact_id=f"sha256:{digest}",
            artifact_key=artifact_key,
            file_name=path.name,
            mime_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            size_bytes=path.stat().st_size,
            sha256=digest,
            drawer_section=drawer_section,
            source_type="skill_script",
            registration_status=RegistrationStatus.PENDING,
        )
        artifacts.append(artifact)
        sources.append(
            ProviderArtifactSource(
                artifact_id=artifact.artifact_id,
                local_path=str(path),
            )
        )
    missing_required = [
        item.artifact_key
        for item in context.artifact_specs
        if item.required and item.artifact_key not in matched_keys
    ]
    return artifacts, sources, missing_required


def _store_raw_execution(
    context: ProviderContext,
    payload: dict[str, Any],
) -> ResourceRef | None:
    try:
        from src.services.resource_store import (
            persist_resource_bytes,
            resource_scope_from_provider_context,
            serialize_resource_payload,
        )

        content = serialize_resource_payload(payload)
        return persist_resource_bytes(
            scope=resource_scope_from_provider_context(context),
            content_bytes=content,
            kind="skill_execution_trace",
            source_type="tool_raw_result",
            file_name=f"{context.capability_id.replace(':', '_')}_execution.json",
            media_type="application/vnd.evoengine.skill-execution+json",
            tool_name=context.capability_id,
            metadata={"capability_id": context.capability_id},
        )
    except Exception:  # pragma: no cover - durable trace is best effort
        return None


class SkillScriptProvider:
    provider_type = ProviderType.SKILL_SCRIPT

    def __init__(self, *, python_executable: str | None = None, rscript_executable: str = "Rscript") -> None:
        self._python_executable = python_executable or os.getenv("PYTHON") or sys.executable
        self._rscript_executable = rscript_executable

    async def invoke(
        self,
        *,
        arguments: dict[str, Any],
        input_refs: list[dict[str, Any]],
        context: ProviderContext,
    ) -> ProviderInvocationResult:
        try:
            skill_dir, script_path = _safe_script_path(context.provider_ref)
        except (ValueError, FileNotFoundError):
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Skill script is unavailable",
                error=CapabilityError(
                    kind=CapabilityErrorKind.PROVIDER_UNAVAILABLE,
                    message="Skill script is unavailable",
                ),
            )
        interpreter = self._python_executable if script_path.suffix.lower() == ".py" else self._rscript_executable
        requested_output_dir = str(context.metadata.get("artifact_output_dir") or "").strip()
        output_dir = (
            Path(requested_output_dir).resolve()
            if requested_output_dir
            else Path(tempfile.mkdtemp(prefix="evo-capability-skill-"))
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_snapshot = _snapshot_output_dir(output_dir)
        encoded_arguments = json.dumps(arguments, ensure_ascii=False)
        encoded_input_refs = json.dumps(input_refs, ensure_ascii=False)
        encoded_materialized_inputs = json.dumps(
            context.metadata.get("materialized_inputs") or [],
            ensure_ascii=False,
        )
        env = dict(os.environ)
        repo_root = Path(__file__).resolve().parents[3]
        existing_pythonpath = str(env.get("PYTHONPATH") or "").strip()
        env["PYTHONPATH"] = (
            str(repo_root)
            if not existing_pythonpath
            else os.pathsep.join([str(repo_root), existing_pythonpath])
        )
        env.update(
            {
                "EVO_SCRIPT_ARGS": encoded_arguments,
                "EVO_SKILL_ARGS": encoded_arguments,
                "EVO_CAPABILITY_INPUT_REFS": encoded_input_refs,
                "EVO_CAPABILITY_INPUTS": encoded_materialized_inputs,
                "EVO_SKILL_OUTPUT_DIR": str(output_dir),
                "EVO_ARTIFACT_OUTPUT_DIR": str(output_dir),
                "EVO_REQUEST_ID": context.request_id,
                "EVO_TRANSPORT_REQUEST_ID": context.transport_request_id,
            }
        )
        if context.project_id:
            env["EVO_PROJECT_ID"] = context.project_id
        if context.conversation_id:
            env["EVO_CONVERSATION_ID"] = context.conversation_id
        if context.user_id is not None:
            env["EVO_USER_ID"] = str(context.user_id)
        try:
            process = await asyncio.create_subprocess_exec(
                interpreter,
                str(script_path),
                cwd=str(skill_dir),
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Skill runtime is unavailable",
                error=CapabilityError(
                    kind=CapabilityErrorKind.PROVIDER_UNAVAILABLE,
                    message="Skill runtime is unavailable",
                ),
            )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=context.timeout_seconds,
            )
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.communicate()
            raise
        stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        execution_trace = {
            "capability_id": context.capability_id,
            "capability_version": context.capability_version,
            "script_name": script_path.name,
            "returncode": process.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
        }
        trace_resource: ResourceRef | None = None

        def _failure_trace_ref() -> str | None:
            """Persist diagnostic stdout/stderr only when execution is unusable."""

            nonlocal trace_resource
            if trace_resource is None:
                trace_resource = _store_raw_execution(context, execution_trace)
            return (
                trace_resource.resource_id
                if trace_resource is not None
                else None
            )

        if process.returncode != 0:
            raw_ref = _failure_trace_ref()
            logger.warning(
                "skill_script_failed capability_id=%s returncode=%s stderr_chars=%d",
                context.capability_id,
                process.returncode,
                len(stderr_text),
            )
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Skill script execution failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.EXECUTION_ERROR,
                    message="Skill script execution failed",
                    details_ref=raw_ref,
                ),
                raw_ref=raw_ref,
            )
        try:
            payload = json.loads(stdout_text)
        except (TypeError, ValueError):
            raw_ref = _failure_trace_ref()
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Skill script output contract validation failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                    message="Skill script stdout must contain one JSON object",
                    details_ref=raw_ref,
                ),
                raw_ref=raw_ref,
            )
        if not isinstance(payload, dict):
            raw_ref = _failure_trace_ref()
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Skill script output contract validation failed",
                error=CapabilityError(
                    kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                    message="Skill script stdout must contain one JSON object",
                    details_ref=raw_ref,
                ),
                raw_ref=raw_ref,
            )
        try:
            capability_outcome = _capability_outcome_from_payload(payload)
        except Exception:
            raw_ref = _failure_trace_ref()
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Skill script capability outcome contract validation failed",
                data=payload,
                error=CapabilityError(
                    kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                    message="Skill script capability_outcome is invalid",
                    details_ref=raw_ref,
                ),
                raw_ref=raw_ref,
            )
        if payload.get("ok") is False:
            raw_ref = _failure_trace_ref()
            raw_error_kind = str(payload.get("error_kind") or "").strip()
            try:
                error_kind = CapabilityErrorKind(raw_error_kind)
            except ValueError:
                error_kind = CapabilityErrorKind.EXECUTION_ERROR
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary=str(payload.get("summary") or "Skill script execution failed"),
                data=payload,
                error=CapabilityError(
                    kind=error_kind,
                    message="Skill script reported failure",
                    details_ref=raw_ref,
                ),
                capability_outcome=capability_outcome,
                raw_ref=raw_ref,
            )
        raw_evidence_refs = payload.get("evidence_refs") or []
        if not isinstance(raw_evidence_refs, list):
            raw_ref = _failure_trace_ref()
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Skill script evidence contract validation failed",
                data=payload,
                error=CapabilityError(
                    kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                    message="Skill script evidence_refs must contain an array",
                    details_ref=raw_ref,
                ),
                raw_ref=raw_ref,
            )
        try:
            evidence_refs = [EvidenceRef.model_validate(item) for item in raw_evidence_refs]
        except Exception:
            raw_ref = _failure_trace_ref()
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Skill script evidence contract validation failed",
                data=payload,
                error=CapabilityError(
                    kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                    message="Skill script evidence_refs contain invalid entries",
                    details_ref=raw_ref,
                ),
                raw_ref=raw_ref,
            )
        artifacts, artifact_sources, missing_required = _artifact_refs(
            output_dir,
            context,
            output_snapshot,
        )
        if missing_required:
            raw_ref = _failure_trace_ref()
            return ProviderInvocationResult(
                status=CapabilityStatus.FAILED,
                summary="Skill script artifact contract validation failed",
                data=payload,
                artifacts=artifacts,
                error=CapabilityError(
                    kind=CapabilityErrorKind.CONTRACT_VIOLATION,
                    message="required Skill artifacts are missing: " + ", ".join(missing_required),
                    details_ref=raw_ref,
                ),
                raw_ref=raw_ref,
                artifact_sources=artifact_sources,
            )
        from src.services.resource_store import (
            persist_tool_result_resource,
            resource_scope_from_provider_context,
        )
        from src.services.tool_machine_projection import project_machine_result

        result_resource = persist_tool_result_resource(
            scope=resource_scope_from_provider_context(context),
            tool_name=context.capability_id,
            value=payload,
            arguments=arguments,
            capability_version=context.capability_version,
        )
        # A successful Skill creates and exposes exactly one business-result
        # ResourceRef.  A trace containing the same stdout is only useful when
        # execution or contract validation fails, so success does not create
        # that duplicate resource at all.
        resources = [result_resource] if result_resource is not None else []
        machine = project_machine_result(payload, resources=resources)
        source_candidate = (
            build_source_candidate_ref(
                capability_id=context.capability_id,
                capability_version=context.capability_version,
                result_resource_id=result_resource.resource_id,
                evidence_policy=context.evidence_policy,
            )
            if result_resource is not None
            else None
        )
        return ProviderInvocationResult(
            status=CapabilityStatus.SUCCEEDED,
            summary=str(payload.get("summary") or "Skill script completed"),
            data=machine.data,
            contract_data=payload,
            source_candidates=[source_candidate] if source_candidate is not None else [],
            resources=list(machine.resources),
            complete=machine.complete,
            has_more=machine.has_more,
            cursor=machine.cursor,
            artifacts=artifacts,
            evidence_refs=evidence_refs,
            capability_outcome=capability_outcome,
            raw_ref=machine.raw_ref,
            artifact_sources=artifact_sources,
        )
