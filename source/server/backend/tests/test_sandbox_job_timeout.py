from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from sqlmodel import SQLModel
from sqlmodel import Session, select

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))


def _load_database_module(monkeypatch, db_name: str):
    db_path = BACKEND_ROOT / db_name
    db_path.unlink(missing_ok=True)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    sys.modules.pop("database", None)
    SQLModel.metadata.clear()
    spec = importlib.util.spec_from_file_location("database_under_test_timeout", BACKEND_ROOT / "database.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _load_sandbox_service_module(monkeypatch, database_module):
    fake_services = types.ModuleType("services")
    fake_services.compute_point_service = SimpleNamespace(consume_for_sandbox_job=lambda *_args, **_kwargs: {"ledger_id": 1})
    fake_services.wallet_service = SimpleNamespace(
        get_wallet_overview=lambda *_args, **_kwargs: {"total_balance": 9999},
        to_decimal=lambda value: value,
    )
    fake_services.sandbox_pricing_service = SimpleNamespace(
        estimate_job_cost=lambda *_args, **_kwargs: {"resource_usage_display": "", "amount": 0},
        resolve_final_billing=lambda *_args, **_kwargs: {
            "amount": 0,
            "resource_usage_display": "",
            "billing_rule_text": "",
        },
        build_app_pricing=lambda *_args, **_kwargs: ({}, {}),
    )
    fake_services.asset_upload_service = SimpleNamespace(
        upload_asset_bytes=lambda *args, **kwargs: None,
        validate_upload_folder=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "database", database_module)
    monkeypatch.setitem(sys.modules, "services", fake_services)

    fake_utils = types.ModuleType("utils")
    fake_cos_client = types.ModuleType("utils.cos_client")
    fake_cos_client.cos_helper = SimpleNamespace(get_file=lambda *args, **kwargs: None, upload_file=lambda *args, **kwargs: None)
    fake_posthog_client = types.ModuleType("utils.posthog_client")
    fake_posthog_client.capture = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "utils", fake_utils)
    monkeypatch.setitem(sys.modules, "utils.cos_client", fake_cos_client)
    monkeypatch.setitem(sys.modules, "utils.posthog_client", fake_posthog_client)

    module_name = "sandbox_service_under_test_timeout"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, BACKEND_ROOT / "services" / "sandbox_service.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def _create_job(database_module, *, job_id: str, status: str, max_run_duration_seconds: int = 300):
    database_module.SQLModel.metadata.create_all(database_module.engine)
    with Session(database_module.engine) as db:
        app = db.exec(select(database_module.SandboxApp).where(database_module.SandboxApp.app_id == "blast")).first()
        if app is None:
            db.add(
                database_module.SandboxApp(
                    app_id="blast",
                    name="BLAST",
                    resource_type="cpu",
                    min_compute_points=1,
                )
            )
        job = database_module.SandboxJob(
            job_id=job_id,
            user_id=7,
            project_id="project_demo",
            app_id="blast",
            app_name="BLAST",
            job_name="超时测试",
            resource_type="cpu",
            status=status,
            billing_status=database_module.SandboxBillingStatus.PENDING,
            max_run_duration_seconds=max_run_duration_seconds,
        )
        db.add(job)
        db.commit()


def test_update_job_timeout_allows_queued_job(monkeypatch):
    database_module = _load_database_module(monkeypatch, "test_sandbox_job_timeout_queued.db")
    sandbox_service = _load_sandbox_service_module(monkeypatch, database_module)
    _create_job(database_module, job_id="job_timeout_queued", status=database_module.SandboxJobStatus.QUEUED)

    with Session(database_module.engine) as db:
      result = sandbox_service.update_job_timeout(
          db,
          user_id=7,
          job_id="job_timeout_queued",
          max_run_duration_seconds=900,
      )
      db.commit()
      job = db.exec(select(database_module.SandboxJob).where(database_module.SandboxJob.job_id == "job_timeout_queued")).first()

    assert result["job"]["max_run_duration_seconds"] == 900
    assert job is not None
    assert job.max_run_duration_seconds == 900


def test_update_job_timeout_allows_running_job(monkeypatch):
    database_module = _load_database_module(monkeypatch, "test_sandbox_job_timeout_running.db")
    sandbox_service = _load_sandbox_service_module(monkeypatch, database_module)
    _create_job(database_module, job_id="job_timeout_running", status=database_module.SandboxJobStatus.RUNNING)

    with Session(database_module.engine) as db:
        result = sandbox_service.update_job_timeout(
            db,
            user_id=7,
            job_id="job_timeout_running",
            max_run_duration_seconds=1200,
        )
        db.commit()

    assert result["job"]["max_run_duration_seconds"] == 1200


def test_update_job_timeout_rejects_terminal_job(monkeypatch):
    database_module = _load_database_module(monkeypatch, "test_sandbox_job_timeout_terminal.db")
    sandbox_service = _load_sandbox_service_module(monkeypatch, database_module)
    _create_job(database_module, job_id="job_timeout_terminal", status=database_module.SandboxJobStatus.SUCCEEDED)

    with Session(database_module.engine) as db:
        with pytest.raises(HTTPException) as exc_info:
            sandbox_service.update_job_timeout(
                db,
                user_id=7,
                job_id="job_timeout_terminal",
                max_run_duration_seconds=900,
            )

    assert exc_info.value.status_code == 400
    assert "仅排队中或运行中的任务可以修改超时设置" in str(exc_info.value.detail)


def test_update_job_timeout_rejects_invalid_seconds(monkeypatch):
    database_module = _load_database_module(monkeypatch, "test_sandbox_job_timeout_invalid.db")
    sandbox_service = _load_sandbox_service_module(monkeypatch, database_module)
    _create_job(database_module, job_id="job_timeout_invalid", status=database_module.SandboxJobStatus.QUEUED)

    with Session(database_module.engine) as db:
        with pytest.raises(HTTPException) as too_small:
            sandbox_service.update_job_timeout(
                db,
                user_id=7,
                job_id="job_timeout_invalid",
                max_run_duration_seconds=30,
            )
        with pytest.raises(HTTPException) as too_large:
            sandbox_service.update_job_timeout(
                db,
                user_id=7,
                job_id="job_timeout_invalid",
                max_run_duration_seconds=200000,
            )

    assert too_small.value.status_code == 400
    assert "不能小于 60 秒" in str(too_small.value.detail)
    assert too_large.value.status_code == 400
    assert "不能超过 172800 秒" in str(too_large.value.detail)
