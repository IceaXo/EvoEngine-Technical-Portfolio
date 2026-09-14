from __future__ import annotations

import http.client
import importlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest

SERVER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SERVER_ROOT))

from sandbox_workers.durable_job_protocol import (
    DurableJobConflict,
    DurableJobProtocol,
)


def _wait_for_terminal(protocol: DurableJobProtocol, job_id: str) -> dict:
    deadline = time.time() + 3
    while time.time() < deadline:
        state = protocol.status(job_id)
        assert state is not None
        if state["status"] in {"completed", "failed"}:
            return state
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not become terminal")


def test_protocol_persists_terminal_result_and_executes_job_id_once(tmp_path):
    started = threading.Event()
    release = threading.Event()
    calls: list[tuple[str, dict]] = []

    def runner(job_id: str, params: dict) -> list[dict]:
        calls.append((job_id, params))
        started.set()
        assert release.wait(timeout=2)
        return [
            {
                "artifact_key": "result",
                "mime_type": "application/json",
                "content": "x" * 100_000,
            }
        ]

    protocol = DurableJobProtocol(tmp_path / "state", runner, worker_name="test")
    first = protocol.submit("job-1", {"sequence": "ACGT"})
    assert first["status"] == "pending"
    assert started.wait(timeout=1)

    duplicate = protocol.submit("job-1", {"sequence": "ACGT"})
    assert duplicate["status"] in {"running", "completed"}
    with pytest.raises(DurableJobConflict):
        protocol.submit("job-1", {"sequence": "TGCA"})

    release.set()
    completed = _wait_for_terminal(protocol, "job-1")
    assert completed["status"] == "completed"
    assert len(completed["artifacts"][0]["content"]) == 100_000
    assert calls == [("job-1", {"sequence": "ACGT"})]

    # A new protocol instance represents a worker process restart.  Terminal
    # state is read from disk and the runner is not entered again.
    after_restart = DurableJobProtocol(
        tmp_path / "state",
        lambda *_args: pytest.fail("completed job must not run again"),
        worker_name="test",
    )
    assert after_restart.submit("job-1", {"sequence": "ACGT"}) == completed


def test_protocol_fails_interrupted_state_instead_of_replaying_it(tmp_path):
    state_dir = tmp_path / "state"
    seed = DurableJobProtocol(state_dir, lambda *_args: [], worker_name="test")
    params = {"input": "value"}
    seed._write_state(  # construct the state left by a killed worker process
        {
            "job_id": "interrupted",
            "status": "running",
            "message": "working",
            "request_fingerprint": seed._request_fingerprint(params),
            "params": params,
        }
    )
    calls = 0

    def runner(*_args):
        nonlocal calls
        calls += 1
        return []

    recovered = DurableJobProtocol(state_dir, runner, worker_name="test")
    state = recovered.status("interrupted")
    assert state is not None
    assert state["status"] == "failed"
    assert "不会自动重跑" in state["message"]
    assert recovered.submit("interrupted", params)["status"] == "failed"
    assert calls == 0


def test_protocol_expires_only_terminal_results(tmp_path):
    protocol = DurableJobProtocol(
        tmp_path / "state",
        lambda *_args: [{"artifact_key": "result", "content": "ok"}],
        worker_name="test",
        terminal_retention_seconds=10,
    )
    protocol.submit("terminal", {})
    completed = _wait_for_terminal(protocol, "terminal")
    assert completed["finished_at"] > 0

    terminal_path = protocol._state_path("terminal")
    terminal_state = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal_state["finished_at"] = time.time() - 11
    protocol._write_state(terminal_state)

    active_state = {
        "job_id": "active",
        "status": "running",
        "message": "working",
        "request_fingerprint": protocol._request_fingerprint({}),
        "params": {},
        "finished_at": time.time() - 100,
    }
    protocol._write_state(active_state)
    active_path = protocol._state_path("active")

    protocol._cleanup_expired_terminal_jobs(force=True)
    assert not terminal_path.exists()
    assert active_path.exists()


def _request(server, method: str, path: str, payload: dict | None = None):
    host, port = server.server_address
    connection = http.client.HTTPConnection(host, port, timeout=2)
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    result = response.status, dict(response.getheaders()), json.loads(raw or b"{}")
    connection.close()
    return result


@pytest.mark.parametrize(
    "module_name",
    [
        "sandbox_workers.ice.ice_server",
        "sandbox_workers.crispor.crispor_server",
    ],
)
def test_worker_http_accepts_then_polls_and_deduplicates(
    tmp_path,
    monkeypatch,
    module_name: str,
):
    module = importlib.import_module(module_name)
    calls: list[tuple[str, dict]] = []

    def runner(job_id: str, params: dict) -> list[dict]:
        calls.append((job_id, params))
        return [
            {
                "artifact_key": "result",
                "label": "result",
                "mime_type": "text/plain",
                "content": "ok",
            }
        ]

    monkeypatch.setattr(module, "WORKDIR", tmp_path / module_name.rsplit(".", 2)[1])
    monkeypatch.setattr(module, "run_job", runner)
    monkeypatch.setattr(module, "_PROTOCOL", None)

    server = module.ThreadingHTTPServer(("127.0.0.1", 0), module.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        payload = {"job_id": "durable-job", "params": {"input": "value"}}
        status, headers, accepted = _request(server, "POST", "/run", payload)
        assert status == 202
        assert accepted["job_id"] == "durable-job"
        assert accepted["status"] in {"pending", "running", "completed"}
        assert headers["Location"] == "/status?job_id=durable-job"

        deadline = time.time() + 2
        while True:
            status, _headers, result = _request(
                server,
                "GET",
                "/status?job_id=durable-job",
            )
            assert status == 200
            if result["status"] == "completed":
                break
            assert result["status"] in {"pending", "running"}
            if time.time() >= deadline:
                raise AssertionError("worker status did not become completed")
            time.sleep(0.01)
        assert result["artifacts"][0]["artifact_key"] == "result"

        status, _headers, duplicate = _request(server, "POST", "/run", payload)
        assert status == 202
        assert duplicate["status"] == "completed"
        assert calls == [("durable-job", {"input": "value"})]

        changed = {"job_id": "durable-job", "params": {"input": "changed"}}
        status, _headers, conflict = _request(server, "POST", "/run", changed)
        assert status == 409
        assert conflict["job_id"] == "durable-job"

        # Reconstructing the protocol still serves the durable result and
        # cannot execute the job a second time.
        monkeypatch.setattr(module, "_PROTOCOL", None)
        status, _headers, persisted = _request(
            server,
            "GET",
            "/status?job_id=durable-job",
        )
        assert status == 200
        assert persisted == result
        assert len(calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
