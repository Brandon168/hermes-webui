"""Model manager backend: gateway metadata cache + enriched catalog (#7507/04).

Covers the patch-04 server surface:
- _manager_bare_id strips only the @provider: routing prefix (never variant
  suffixes: -fast/-pro/-mini/date stay distinct).
- _manager_meta_for joins exact-id only (no suffix inheritance).
- refresh_models_meta trims the public feed, enriches missing dates from
  fallback sources, and persists models_meta + refresh timestamps.
- get_model_manager_catalog enriches rows with hidden flags + metadata and
  never drops rows (UI applies the non-chat rule explicitly).
- save_settings accepts/replaces models_meta wholesale.
"""

import time

import api.config as cfg


def _isolate(tmp_path, monkeypatch):
    import api.config as c

    old_settings = getattr(c, "SETTINGS_FILE", None)
    fake = tmp_path / "settings.json"
    fake.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(c, "SETTINGS_FILE", fake)
    old_cfg = dict(c.cfg)
    try:
        c.invalidate_models_cache()
    except Exception:
        pass
    yield
    c.cfg.clear()
    c.cfg.update(old_cfg)
    if old_settings is not None:
        monkeypatch.setattr(c, "SETTINGS_FILE", old_settings)
    try:
        c.invalidate_models_cache()
    except Exception:
        pass


import pytest

_iso = pytest.fixture(autouse=True)(_isolate)


def test_bare_id_strips_only_provider_prefix():
    assert cfg._manager_bare_id("@vercel-brandon-pro:openai/gpt-5.6-luna") == "openai/gpt-5.6-luna"
    # variant suffixes are preserved, never folded
    assert cfg._manager_bare_id("openai/gpt-5.6-luna-fast") == "openai/gpt-5.6-luna-fast"
    assert cfg._manager_bare_id("openai/gpt-5.6-luna-pro") == "openai/gpt-5.6-luna-pro"
    assert cfg._manager_bare_id("deepseek/deepseek-v4-flash-0731") == "deepseek/deepseek-v4-flash-0731"


def test_meta_join_is_exact_id_only():
    meta = {"openai/gpt-5.6-luna": {"type": "language", "released": 1780000000}}
    assert cfg._manager_meta_for("openai/gpt-5.6-luna", meta)["released"] == 1780000000
    # routed form resolves to the same bare id
    assert cfg._manager_meta_for("@vercel-brandon-pro:openai/gpt-5.6-luna", meta)["released"] == 1780000000
    # variants do NOT inherit the base model's metadata
    assert cfg._manager_meta_for("openai/gpt-5.6-luna-fast", meta) is None
    assert cfg._manager_meta_for("openai/gpt-5.6-luna-pro", meta) is None


def test_refresh_trims_and_persists_feed(monkeypatch):
    import json as _json

    feed = {"data": [
        {"id": "openai/gpt-5.6-luna", "type": "language", "released": 1780000000,
         "created": 1779000000, "name": "Luna",
         "pricing": {"input": "0.000002", "output": "0.000012"},
         "tags": ["reasoning", "tool-use"]},
        {"id": "typesafe-ai/jev", "type": "evaluation", "released": 1789430400,
         "pricing": {}, "tags": None},
        {"id": "", "type": "language"},
        "not-a-dict",
    ]}

    class _Resp:
        def read(self):
            return _json.dumps(feed).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    import urllib.request as _urlreq
    monkeypatch.setattr(_urlreq, "urlopen", lambda req, timeout=None: _Resp())
    # An empty inventory needs no fallback sources; prove none is fetched.
    monkeypatch.setattr(cfg, "get_available_models", lambda **kwargs: {"groups": []})
    monkeypatch.setattr(cfg, "_fetch_modelsdev_release_dates",
                        lambda **kwargs: pytest.fail("models.dev fetched with nothing to resolve"))
    monkeypatch.setattr(cfg, "_fetch_openrouter_created_dates",
                        lambda **kwargs: pytest.fail("openrouter fetched with nothing to resolve"))
    out = cfg.refresh_models_meta()
    assert out["ok"] is True and out["models"] == 2 and out["fallback_dates"] == 0
    stored = cfg._load_gateway_models_meta()
    luna = stored["openai/gpt-5.6-luna"]
    assert luna["type"] == "language" and luna["released"] == 1780000000
    assert luna["released_source"] == "gateway"
    assert luna["pricing"] == {"input": "0.000002", "output": "0.000012"}
    assert luna["tags"] == ["reasoning", "tool-use"]
    assert stored["typesafe-ai/jev"]["type"] == "evaluation"
    # unknown keys are dropped by save_settings validation
    raw = _json.loads(cfg.SETTINGS_FILE.read_text(encoding="utf-8"))
    assert set(raw["models_meta"]["openai/gpt-5.6-luna"]) <= {
        "type", "released", "released_source", "created", "pricing", "tags", "name"}
    # refresh timestamps persist and drive the TTL auto-refresh
    assert int(raw["models_meta_refreshed_at"]) > 0
    assert int(raw["models_meta_refresh_attempted_at"]) > 0


def test_manager_catalog_enriches_without_dropping(monkeypatch):
    monkeypatch.setattr(cfg, "get_available_models", lambda **kwargs: {
        "groups": [{
            "provider": "V", "provider_id": "vercel-brandon-pro",
            "models": [
                {"id": "openai/gpt-5.6-luna", "label": "Luna"},
                {"id": "openai/gpt-5.6-luna-fast", "label": "Luna fast"},
            ],
            "extra_models": [{"id": "openai/gpt-image-1.5", "label": "img"}],
        }],
    })
    cfg.save_settings({"models_hidden": {"vercel-brandon-pro": ["openai/gpt-5.6-luna"]}})
    now = int(time.time())
    cfg.save_settings({"models_meta": {
        "openai/gpt-5.6-luna": {"type": "language", "released": 1780000000},
        "openai/gpt-image-1.5": {"type": "image", "released": 1781000000},
    }, "models_meta_refreshed_at": now, "models_meta_refresh_attempted_at": now})
    refreshes = []
    monkeypatch.setattr(cfg, "refresh_models_meta", lambda **kwargs: refreshes.append(1))
    cat = cfg.get_model_manager_catalog()
    assert refreshes == []  # a fresh snapshot is never re-fetched
    assert cat["meta_count"] == 2 and cat["meta_fresh"] is True
    rows = cat["groups"][0]["models"]
    # nothing dropped: 2 visible + 1 overflow = 3 rows
    assert len(rows) == 3
    by_id = {r["id"]: r for r in rows}
    assert by_id["openai/gpt-5.6-luna"]["hidden"] is True
    assert by_id["openai/gpt-5.6-luna-fast"]["hidden"] is False
    # fast variant has no metadata of its own (no inheritance)
    assert by_id["openai/gpt-5.6-luna-fast"]["meta"] is None
    # non-chat row is present with its type (UI applies the rule)
    assert by_id["openai/gpt-image-1.5"]["meta"]["type"] == "image"


def test_manager_catalog_refresh_failure_serves_cache(monkeypatch):
    monkeypatch.setattr(cfg, "get_available_models", lambda **kwargs: {"groups": []})
    cfg.save_settings({"models_meta": {"a/b": {"type": "language"}}})

    def _boom(*a, **k):
        raise RuntimeError("net down")
    monkeypatch.setattr(cfg, "refresh_models_meta", _boom)
    cat = cfg.get_model_manager_catalog(refresh=True)
    assert cat["meta_fresh"] is False and cat["meta_count"] == 1


def test_refresh_enriches_missing_dates_from_fallback_sources(monkeypatch):
    monkeypatch.setattr(cfg, "get_available_models", lambda **kwargs: {
        "groups": [{
            "provider": "CC", "provider_id": "commandcode",
            "models": [
                {"id": "xai/grok-4.7", "label": "grok"},
                {"id": "stealth/space-bunny-alpha", "label": "bunny"},
                {"id": "alibaba/qwen3.8-max-prime", "label": "qwen"},
            ],
            "extra_models": [{"id": "cohere/north-mini-code:free", "label": "north"}],
        }],
    })
    # A stored hidden-list id for an undiscovered provider renders as a
    # placeholder row in the manager and must be enriched like any other row.
    cfg.save_settings({"models_hidden": {"somewhere": ["nex-agi/nex-n2.5-pro:free"]}})
    monkeypatch.setattr(cfg, "_fetch_gateway_meta", lambda **kwargs: {
        "alibaba/qwen3.8-max-prime": {"type": "language", "released": 1790100000, "released_source": "gateway"},
    })
    md_calls, or_calls = [], []

    def _md(**kwargs):
        md_calls.append(1)
        return {"grok-4.7": [("xai", 1789000000)], "space-bunny-alpha": [("openrouter", 1788800000)]}

    def _or(**kwargs):
        or_calls.append(1)
        return {
            "north-mini-code": [("cohere/north-mini-code:free", 1788000000)],
            "nex-n2.5-pro": [("nex-agi/nex-n2.5-pro:free", 1787000000)],
        }

    monkeypatch.setattr(cfg, "_fetch_modelsdev_release_dates", _md)
    monkeypatch.setattr(cfg, "_fetch_openrouter_created_dates", _or)
    out = cfg.refresh_models_meta()
    assert out["ok"] is True and out["fallback_dates"] == 4
    stored = cfg._load_gateway_models_meta()
    assert stored["xai/grok-4.7"]["released"] == 1789000000
    assert stored["xai/grok-4.7"]["released_source"] == "models.dev"
    assert stored["stealth/space-bunny-alpha"]["released"] == 1788800000
    assert stored["cohere/north-mini-code:free"]["released"] == 1788000000
    assert stored["cohere/north-mini-code:free"]["released_source"] == "openrouter"
    assert stored["nex-agi/nex-n2.5-pro:free"]["released"] == 1787000000
    # gateway-dated rows are left untouched
    assert stored["alibaba/qwen3.8-max-prime"]["released_source"] == "gateway"
    # each fallback source is consulted at most once per refresh
    assert md_calls == [1] and or_calls == [1]


def test_fallback_pickers_prefer_canonical_sources():
    # models.dev: the first-party provider wins over the majority; ties resolve to the earliest date
    assert cfg._pick_modelsdev_date([("nano-gpt", 1), ("zai", 9), ("kilo", 1)], "zai") == 9
    assert cfg._pick_modelsdev_date([("a", 5), ("b", 5), ("c", 7)], None) == 5
    assert cfg._pick_modelsdev_date([], "zai") is None
    # OpenRouter: exact catalog id first, then vendor prefix, else the earliest record
    assert cfg._pick_openrouter_date("x/y", [("a/y", 9), ("x/y", 2)], "a") == 2
    assert cfg._pick_openrouter_date("xai/grok-4.7", [("x-ai/grok-4.7", 7), ("nano-gpt/x", 3)], "x-ai") == 7
    assert cfg._pick_openrouter_date("x/y", [("a/y", 9), ("b/y", 4)], None) == 4


def test_iso_date_conversion_and_slug_lookup():
    import calendar as _cal
    assert cfg._iso_date_to_epoch("2026-09-21") == int(_cal.timegm(time.strptime("2026-09-21", "%Y-%m-%d")))
    assert cfg._iso_date_to_epoch(None) is None and cfg._iso_date_to_epoch("garbage") is None
    assert cfg._source_slug("Cohere/North-Mini-Code:free") == "north-mini-code"
    assert cfg._source_slug("stealth/space-bunny-alpha") == "space-bunny-alpha"


def test_manager_catalog_auto_refreshes_when_stale(monkeypatch):
    monkeypatch.setattr(cfg, "get_available_models", lambda **kwargs: {"groups": []})
    cfg.save_settings({"models_meta": {"a/b": {"type": "language", "released": 1}}})
    stale = int(time.time()) - cfg._MANAGER_META_TTL_SECONDS - 60
    cfg.save_settings({"models_meta_refreshed_at": stale, "models_meta_refresh_attempted_at": 0})
    calls = []

    def _refresh(**kwargs):
        calls.append(1)
        stamp = int(time.time())
        cfg.save_settings({"models_meta_refreshed_at": stamp, "models_meta_refresh_attempted_at": stamp})
        return {"ok": True, "models": 1}

    monkeypatch.setattr(cfg, "refresh_models_meta", _refresh)
    cat = cfg.get_model_manager_catalog()
    assert calls == [1] and cat["meta_fresh"] is True
    cfg.get_model_manager_catalog()
    assert calls == [1]  # snapshot is fresh now; no second fetch


def test_manager_catalog_backs_off_after_failed_auto_refresh(monkeypatch):
    monkeypatch.setattr(cfg, "get_available_models", lambda **kwargs: {"groups": []})
    cfg.save_settings({"models_meta": {"a/b": {"type": "language", "released": 1}}})
    stale = int(time.time()) - cfg._MANAGER_META_TTL_SECONDS - 60
    cfg.save_settings({"models_meta_refreshed_at": stale, "models_meta_refresh_attempted_at": 0})
    calls = []

    def _boom(**kwargs):
        calls.append(1)
        cfg.save_settings({"models_meta_refresh_attempted_at": int(time.time())})
        raise OSError("offline")

    monkeypatch.setattr(cfg, "refresh_models_meta", _boom)
    cat = cfg.get_model_manager_catalog()
    assert calls == [1] and cat["meta_fresh"] is False and cat["meta_count"] == 1
    cfg.get_model_manager_catalog()
    assert calls == [1]  # still backing off; no hammering


def test_refresh_failure_records_attempt_and_keeps_snapshot(monkeypatch):
    cfg.save_settings({"models_meta": {"keep/me": {"type": "language", "released": 1}}})
    monkeypatch.setattr(cfg, "_fetch_gateway_meta",
                        lambda **kwargs: (_ for _ in ()).throw(OSError("down")))
    with pytest.raises(OSError):
        cfg.refresh_models_meta()
    raw = cfg._read_raw_settings_file()
    assert raw["models_meta"] == {"keep/me": {"type": "language", "released": 1}}
    assert int(raw["models_meta_refresh_attempted_at"]) > 0
    assert int(raw.get("models_meta_refreshed_at") or 0) == 0


def test_fallback_fetchers_send_user_agent(monkeypatch):
    # models.dev answers 403 to the default urllib User-Agent; both fallback
    # fetchers must identify themselves to keep working.
    import urllib.request as _urlreq

    class _Resp:
        def read(self):
            return b"{}"
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    seen = []

    def _capture(req, timeout=None):
        seen.append(req)
        return _Resp()

    monkeypatch.setattr(_urlreq, "urlopen", _capture)
    cfg._fetch_modelsdev_release_dates()
    cfg._fetch_openrouter_created_dates()
    assert len(seen) == 2
    for req in seen:
        assert req.get_header("User-agent") == cfg._SOURCE_USER_AGENT
