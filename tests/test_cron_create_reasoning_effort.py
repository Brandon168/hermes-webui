"""Cron create/update must carry the per-job ``reasoning_effort`` pin end to end.

The WebUI cron form offered model/provider overrides while the Agent's per-job
``reasoning_effort`` field (``cron.jobs.create_job``/``update_job``) was only
reachable from the CLI. This regression pins the API half of the fix:

1. ``api/routes.py::_handle_cron_create`` forwards ``reasoning_effort`` to
   ``create_job`` (None when absent or empty, matching model/provider).
2. ``api/routes.py::_handle_cron_update`` treats the field like model/provider:
   a value passes through, null/empty clears the override.
3. ``static/panels.js::_cronReasoningEffortOptions`` renders the Agent's effort
   vocabulary and keeps an unknown stored value selectable instead of dropping
   the pin on the next edit.
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PANELS_JS = (REPO / "static" / "panels.js").read_text(encoding="utf-8")


class _JSONHandler:
    def __init__(self):
        self.status = None
        self.headers = {}
        self.response_headers = []
        self.wfile = io.BytesIO()

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.response_headers.append((key, value))

    def end_headers(self):
        pass


def _payload(handler):
    return json.loads(handler.wfile.getvalue().decode("utf-8"))


def _install_cron_jobs_stub(monkeypatch, *, create_fn=None, update_fn=None):
    cron_pkg = types.ModuleType("cron")
    cron_pkg.__path__ = []
    cron_jobs = types.ModuleType("cron.jobs")
    cron_jobs.create_job = create_fn or (lambda **kwargs: {"id": "stub", **kwargs})
    cron_jobs.update_job = update_fn or (lambda job_id, updates: {"id": job_id, **updates})
    monkeypatch.setitem(sys.modules, "cron", cron_pkg)
    monkeypatch.setitem(sys.modules, "cron.jobs", cron_jobs)


def _create_body(**extra):
    body = {"prompt": "standalone test prompt", "schedule": "every day at 5pm"}
    body.update(extra)
    return body


# --- Backend: create forwards reasoning_effort ------------------------------


def test_cron_create_forwards_reasoning_effort(monkeypatch):
    import api.routes as routes

    calls = {}

    def _create(**kwargs):
        calls.update(kwargs)
        return {"id": "new-job", **kwargs}

    _install_cron_jobs_stub(monkeypatch, create_fn=_create)
    handler = _JSONHandler()
    routes._handle_cron_create(handler, _create_body(
        model="gpt-6-luna", provider="commandcode", reasoning_effort="medium"))

    assert handler.status == 200
    assert _payload(handler).get("ok") is True
    assert calls.get("reasoning_effort") == "medium"
    assert calls.get("model") == "gpt-6-luna"
    assert calls.get("provider") == "commandcode"


def test_cron_create_without_effort_sends_none(monkeypatch):
    import api.routes as routes

    calls = {}

    def _create(**kwargs):
        calls.update(kwargs)
        return {"id": "new-job", **kwargs}

    _install_cron_jobs_stub(monkeypatch, create_fn=_create)
    handler = _JSONHandler()
    routes._handle_cron_create(handler, _create_body())
    assert handler.status == 200
    assert calls.get("reasoning_effort") is None


def test_cron_create_empty_effort_sends_none(monkeypatch):
    """Matches the model/provider semantics: empty means no override."""
    import api.routes as routes

    calls = {}

    def _create(**kwargs):
        calls.update(kwargs)
        return {"id": "new-job", **kwargs}

    _install_cron_jobs_stub(monkeypatch, create_fn=_create)
    handler = _JSONHandler()
    routes._handle_cron_create(handler, _create_body(reasoning_effort=""))
    assert handler.status == 200
    assert calls.get("reasoning_effort") is None


# --- Backend: update passes edits through and clears on null/empty ----------


def _updates_seen(monkeypatch, body):
    import api.routes as routes

    seen = {}

    def _update(job_id, updates):
        seen.update(updates)
        return {"id": job_id, **updates}

    _install_cron_jobs_stub(monkeypatch, update_fn=_update)
    handler = _JSONHandler()
    routes._handle_cron_update(handler, {"job_id": "test-job", **body})
    assert handler.status == 200, _payload(handler)
    return seen


def test_cron_update_passes_effort_through(monkeypatch):
    seen = _updates_seen(monkeypatch, {"reasoning_effort": "xhigh"})
    assert seen.get("reasoning_effort") == "xhigh"


def test_cron_update_null_effort_clears(monkeypatch):
    """null must clear the pin (same contract as model/provider)."""
    seen = _updates_seen(monkeypatch, {"reasoning_effort": None})
    assert "reasoning_effort" in seen
    assert seen["reasoning_effort"] is None


def test_cron_update_empty_effort_clears(monkeypatch):
    seen = _updates_seen(monkeypatch, {"reasoning_effort": ""})
    assert "reasoning_effort" in seen
    assert seen["reasoning_effort"] is None


# --- Frontend: option builder executes and preserves the stored pin ---------


def _cron_reasoning_effort_options_source() -> str:
    marker = "function _cronReasoningEffortOptions("
    start = PANELS_JS.find(marker)
    assert start != -1, "_cronReasoningEffortOptions not found in panels.js"
    paren = PANELS_JS.find("(", start)
    depth = 0
    brace = -1
    for idx in range(paren, len(PANELS_JS)):
        ch = PANELS_JS[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                brace = PANELS_JS.find("{", idx)
                break
    assert brace != -1
    depth = 0
    for idx in range(brace, len(PANELS_JS)):
        ch = PANELS_JS[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return PANELS_JS[start:idx + 1]
    raise AssertionError("_cronReasoningEffortOptions body did not terminate")


def _run_options(selected) -> str:
    script = (
        "function esc(s){ return String(s); }\n"
        "function t(k){ return k; }\n"
        f"{_cron_reasoning_effort_options_source()}\n"
        f"process.stdout.write(_cronReasoningEffortOptions({json.dumps(selected)}));\n"
    )
    result = subprocess.run(
        ["node", "-e", script], check=True, capture_output=True, text=True
    )
    return result.stdout


def test_effort_options_cover_agent_vocabulary_and_default():
    html = _run_options("medium")
    for level in ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"):
        assert f'value="{level}"' in html, level
    assert 'value=""' in html
    assert '<option value="medium" selected>' in html


def test_effort_options_preserve_unknown_stored_value():
    """A future Agent level must stay selected rather than reset the pin."""
    html = _run_options("banana")
    assert '<option value="banana" selected>' in html
    assert '<option value="medium" selected>' not in html
