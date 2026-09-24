"""Model manager backend: gateway metadata cache + enriched catalog (#7507/04).

Covers the patch-04 server surface:
- _manager_bare_id strips only the @provider: routing prefix (never variant
  suffixes: -fast/-pro/-mini/date stay distinct).
- _manager_meta_for joins exact-id only (no suffix inheritance).
- refresh_gateway_models_meta trims the public feed and persists models_meta.
- get_model_manager_catalog enriches rows with hidden flags + metadata and
  never drops rows (UI applies the non-chat rule explicitly).
- save_settings accepts/replaces models_meta wholesale.
"""

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
    out = cfg.refresh_gateway_models_meta()
    assert out["ok"] is True and out["models"] == 2
    stored = cfg._load_gateway_models_meta()
    luna = stored["openai/gpt-5.6-luna"]
    assert luna["type"] == "language" and luna["released"] == 1780000000
    assert luna["pricing"] == {"input": "0.000002", "output": "0.000012"}
    assert luna["tags"] == ["reasoning", "tool-use"]
    assert stored["typesafe-ai/jev"]["type"] == "evaluation"
    # unknown keys are dropped by save_settings validation
    raw = _json.loads(cfg.SETTINGS_FILE.read_text(encoding="utf-8"))
    assert set(raw["models_meta"]["openai/gpt-5.6-luna"]) <= {
        "type", "released", "created", "pricing", "tags", "name"}


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
    cfg.save_settings({"models_meta": {
        "openai/gpt-5.6-luna": {"type": "language", "released": 1780000000},
        "openai/gpt-image-1.5": {"type": "image", "released": 1781000000},
    }})
    cat = cfg.get_model_manager_catalog()
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
    monkeypatch.setattr(cfg, "refresh_gateway_models_meta", _boom)
    cat = cfg.get_model_manager_catalog(refresh=True)
    assert cat["meta_fresh"] is False and cat["meta_count"] == 1
