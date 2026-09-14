"""Small, dependency-free durable job protocol for standalone sandbox workers.

The HTTP worker owns request acceptance and execution state.  A request is
persisted before it is acknowledged, and a given ``job_id`` is never executed
twice by the same state store.  This deliberately provides at-most-once
execution: after a worker process restart, an interrupted job is reported as
failed instead of being started again while an orphaned native process may
still exist.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Callable


class DurableJobConflict(ValueError):
    """The caller reused a job id with a different immutable request."""


class DurableJobStateError(RuntimeError):
    """Persisted state is corrupt or does not belong to the requested job."""


class DurableJobProtocol:
    """Persist worker state and run each accepted job at most once."""

    _ACTIVE = {"pending", "running"}
    _TERMINAL = {"completed", "failed"}

    def __init__(
        self,
        state_dir: Path,
        runner: Callable[[str, dict], list[dict]],
        *,
        worker_name: str,
        terminal_retention_seconds: int = 86_400,
    ) -> None:
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.runner = runner
        self.worker_name = str(worker_name)
        self.terminal_retention_seconds = int(terminal_retention_seconds)
        if self.terminal_retention_seconds <= 0:
            raise ValueError("terminal_retention_seconds must be positive")
        self._cleanup_interval_seconds = min(
            300,
            max(1, self.terminal_retention_seconds // 10),
        )
        self._last_cleanup_at = 0.0
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        self._cleanup_expired_terminal_jobs(force=True)
        self._fail_interrupted_jobs()

    @staticmethod
    def _request_fingerprint(params: dict) -> str:
        canonical = json.dumps(
            params,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _state_path(self, job_id: str) -> Path:
        digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()
        return self.state_dir / f"{digest}.json"

    def _read_state(self, job_id: str) -> dict | None:
        path = self._state_path(job_id)
        if not path.exists():
            return None
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise DurableJobStateError(
                f"persisted state for job {job_id!r} is unreadable"
            ) from exc
        if not isinstance(state, dict) or state.get("job_id") != job_id:
            raise DurableJobStateError(
                f"persisted state for job {job_id!r} has invalid identity"
            )
        if state.get("status") not in self._ACTIVE | self._TERMINAL:
            raise DurableJobStateError(
                f"persisted state for job {job_id!r} has invalid status"
            )
        return state

    def _write_state(self, state: dict) -> None:
        path = self._state_path(str(state["job_id"]))
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        encoded = json.dumps(
            state,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            with temp_path.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
            directory_fd = os.open(self.state_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass

    @staticmethod
    def _public_state(state: dict) -> dict:
        public = {
            "job_id": str(state["job_id"]),
            "status": str(state["status"]),
            "message": str(state.get("message") or ""),
        }
        if state.get("status") == "completed":
            public["artifacts"] = state.get("artifacts") or []
        if state.get("status") in DurableJobProtocol._TERMINAL:
            public["finished_at"] = state.get("finished_at")
        return public

    def submit(self, job_id: str, params: dict) -> dict:
        """Durably accept a request and schedule its one allowed execution."""

        fingerprint = self._request_fingerprint(params)
        with self._lock:
            self._cleanup_expired_terminal_jobs()
            existing = self._read_state(job_id)
            if existing is not None:
                if existing.get("request_fingerprint") != fingerprint:
                    raise DurableJobConflict(
                        "job_id 已存在，且本次参数与首次提交不一致"
                    )
                return self._public_state(existing)

            state = {
                "job_id": job_id,
                "status": "pending",
                "message": "任务已接收，等待执行",
                "request_fingerprint": fingerprint,
                "params": params,
            }
            self._write_state(state)
            thread = threading.Thread(
                target=self._execute,
                args=(job_id,),
                name=f"{self.worker_name}-{self._state_path(job_id).stem[:24]}",
                daemon=True,
            )
            self._threads[job_id] = thread
            thread.start()
            return self._public_state(state)

    def status(self, job_id: str) -> dict | None:
        with self._lock:
            self._cleanup_expired_terminal_jobs()
            state = self._read_state(job_id)
            return self._public_state(state) if state is not None else None

    def _execute(self, job_id: str) -> None:
        try:
            with self._lock:
                state = self._read_state(job_id)
                if state is None or state.get("status") != "pending":
                    return
                state["status"] = "running"
                state["message"] = "任务正在执行"
                self._write_state(state)
                params = state.get("params")
            if not isinstance(params, dict):
                raise DurableJobStateError("persisted params must be an object")

            artifacts = self.runner(job_id, params)
            if not isinstance(artifacts, list):
                raise DurableJobStateError("worker runner must return an artifact list")
            terminal = {
                "job_id": job_id,
                "status": "completed",
                "message": "任务完成",
                "request_fingerprint": state["request_fingerprint"],
                "finished_at": time.time(),
                "artifacts": artifacts,
            }
        except Exception as exc:  # noqa: BLE001 - failures become durable state
            traceback.print_exc()
            message = str(getattr(exc, "message", None) or exc or type(exc).__name__)
            with self._lock:
                previous = self._read_state(job_id) or {}
            terminal = {
                "job_id": job_id,
                "status": "failed",
                "message": message[:4000],
                "request_fingerprint": previous.get("request_fingerprint", ""),
                "finished_at": time.time(),
            }
        finally:
            with self._lock:
                if "terminal" in locals():
                    self._write_state(terminal)
                self._threads.pop(job_id, None)

    def _fail_interrupted_jobs(self) -> None:
        """Never replay work whose previous execution outcome is unknown."""

        for path in self.state_dir.glob("*.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if not isinstance(state, dict) or state.get("status") not in self._ACTIVE:
                continue
            job_id = str(state.get("job_id") or "")
            if not job_id or self._state_path(job_id) != path:
                continue
            terminal = {
                "job_id": job_id,
                "status": "failed",
                "message": (
                    "worker 在任务执行期间重启；为避免重复计算，任务不会自动重跑"
                ),
                "request_fingerprint": str(state.get("request_fingerprint") or ""),
                "finished_at": time.time(),
            }
            self._write_state(terminal)

    def _cleanup_expired_terminal_jobs(self, *, force: bool = False) -> None:
        """Bound terminal-result storage without ever touching active jobs."""

        now_monotonic = time.monotonic()
        if (
            not force
            and now_monotonic - self._last_cleanup_at
            < self._cleanup_interval_seconds
        ):
            return
        self._last_cleanup_at = now_monotonic
        expires_before = time.time() - self.terminal_retention_seconds
        removed = False
        for path in self.state_dir.glob("*.json"):
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(state, dict):
                    continue
                status = state.get("status")
                finished_at = float(state.get("finished_at", 0))
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                continue
            if status not in self._TERMINAL or finished_at <= 0:
                continue
            if finished_at > expires_before:
                continue
            try:
                path.unlink()
                removed = True
            except FileNotFoundError:
                pass
        if removed:
            directory_fd = os.open(self.state_dir, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
