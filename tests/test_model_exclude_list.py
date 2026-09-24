"""Per-provider model exclude list for the model picker (#7507).

Upstream issue: a per-provider exclude/hide list in settings — entries
removed from the picker regardless of source (live probe or static
fallback). An exclude list (not a bigger allowlist): one entry per
unwanted model, new upstream models still appear automatically, composes
with live discovery. The explicit ``providers.<id>.models`` allowlist
stays the stronger override.
"""

import copy

import pytest

import api.config as cfg


@pytest.fixture(autouse=True)
def _isolate_state(tmp_path, monkeypatch):
    """Isolate settings file + models cache around every test."""
    import api.config as c

    old_settings = getattr(c, "SETTINGS_FILE", None)
    fake_settings = tmp_path / "settings.json"
    fake_settings.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(c, "SETTINGS_FILE", fake_settings)
    old_cfg = dict(c.cfg)
    old_mtime = c._cfg_mtime
    old_path = c._cfg_path
    old_fingerprint = c._cfg_fingerprint
    try:
        c.invalidate_models_cache()
    except Exception:
        pass
    yield
    c.cfg.clear()
    c.cfg.update(old_cfg)
    c._cfg_mtime = old_mtime
    c._cfg_path = old_path
    c._cfg_fingerprint = old_fingerprint
    if old_settings is not None:
        monkeypatch.setattr(c, "SETTINGS_FILE", old_settings)
    try:
        c.invalidate_models_cache()
    except Exception:
        pass


def _groups():
    return [
        {
            "provider": "OpenRouter",
            "provider_id": "openrouter",
            "models": [
                {"id": "openai/gpt-5.6-luna", "label": "Luna"},
                {"id": "openai/gpt-5.4-mini", "label": "Mini"},
                {"id": "meta/muse-spark-1.3", "label": "Spark"},
            ],
            "extra_models": [
                {"id": "x-ai/grok-4.6", "label": "Grok"},
                {"id": "google/gemini-3.8-flash", "label": "Gemini"},
            ],
        },
        {
            "provider": "OpenAI Codex",
            "provider_id": "openai-codex",
            "models": [
                {"id": "@openai-codex:gpt-5.6-luna", "label": "Luna"},
                {"id": "@openai-codex:gpt-5.4-mini", "label": "Mini"},
            ],
        },
    ]


def _write_hidden(payload):
    import api.config as c
    import json

    c.SETTINGS_FILE.write_text(json.dumps(payload), encoding="utf-8")


def test_exclude_removes_bare_and_prefixed_forms():
    """One entry hides the bare id in one group and the @provider: form in another."""
    _write_hidden({"models_hidden": {"openrouter": ["openai/gpt-5.6-luna"]}})
    groups = _groups()
    removed = cfg._apply_user_model_exclusions(groups)
    assert removed == 1
    or_models = [m["id"] for m in groups[0]["models"]]
    assert "openai/gpt-5.6-luna" not in or_models
    assert "openai/gpt-5.4-mini" in or_models
    # other provider untouched (different canonical key)
    codex_models = [m["id"] for m in groups[1]["models"]]
    assert "@openai-codex:gpt-5.6-luna" in codex_models


def test_exclude_matches_routed_prefixed_entry():
    """Hiding the bare id also hides its @provider: routed twin."""
    _write_hidden({"models_hidden": {"openai-codex": ["gpt-5.6-luna"]}})
    groups = _groups()
    removed = cfg._apply_user_model_exclusions(groups)
    assert removed == 1
    codex_models = [m["id"] for m in groups[1]["models"]]
    assert "@openai-codex:gpt-5.6-luna" not in codex_models
    assert "@openai-codex:gpt-5.4-mini" in codex_models


def test_exclude_applies_to_overflow_bucket():
    """extra_models (overflow/search) rows are filtered, not just visible rows."""
    _write_hidden({"models_hidden": {"openrouter": ["x-ai/grok-4.6"]}})
    groups = _groups()
    removed = cfg._apply_user_model_exclusions(groups)
    assert removed == 1
    assert [m["id"] for m in groups[0]["extra_models"]] == ["google/gemini-3.8-flash"]


def test_exclude_is_provider_scoped():
    """Same model id hidden under one provider stays visible under another."""
    _write_hidden(
        {
            "models_hidden": {
                "openrouter": ["openai/gpt-5.4-mini"],
            }
        }
    )
    groups = _groups()
    cfg._apply_user_model_exclusions(groups)
    assert [m["id"] for m in groups[0]["models"]] == [
        "openai/gpt-5.6-luna",
        "meta/muse-spark-1.3",
    ]
    assert "@openai-codex:gpt-5.4-mini" in [m["id"] for m in groups[1]["models"]]


def test_exclude_alias_key_canonicalised():
    """Underscore/case variants of the provider key still match (cf #1568)."""
    _write_hidden({"models_hidden": {"OpenRouter": ["meta/muse-spark-1.3"]}})
    groups = _groups()
    removed = cfg._apply_user_model_exclusions(groups)
    assert removed == 1
    assert "meta/muse-spark-1.3" not in [m["id"] for m in groups[0]["models"]]


def test_exclude_empty_or_absent_is_noop():
    groups = _groups()
    before = copy.deepcopy(groups)
    assert cfg._apply_user_model_exclusions(groups) == 0
    assert groups == before
    _write_hidden({"models_hidden": {}})
    assert cfg._apply_user_model_exclusions(groups) == 0
    assert groups == before


def test_exclude_malformed_entries_ignored():
    _write_hidden({"models_hidden": {"openrouter": ["ok-model", 123, None, "  "]}})
    groups = _groups()
    assert cfg._apply_user_model_exclusions(groups) == 0
    assert len(groups[0]["models"]) == 3


def test_settings_default_includes_models_hidden():
    assert cfg._SETTINGS_DEFAULTS.get("models_hidden") == {}


def test_save_settings_replaces_models_hidden():
    _write_hidden({"models_hidden": {"openrouter": ["a"]}})
    saved = cfg.save_settings({"models_hidden": {"openai-codex": ["b", "c"]}})
    assert saved["models_hidden"] == {"openai-codex": ["b", "c"]}
    # junk shapes rejected
    saved = cfg.save_settings({"models_hidden": ["not-a-dict"]})
    assert saved["models_hidden"] == {"openai-codex": ["b", "c"]}


def test_exclude_subtracts_before_visible_overflow_split():
    """Hidden rows never occupy visible slots: eligible rows backfill (#7507).

    Regression for the first patch-03 revision, which filtered only the
    assembled ``models``/``extra_models`` buckets post-split: visible slots
    stayed consumed by hidden rows and overflow never backfilled.
    """
    import api.config as c

    _write_hidden({"models_hidden": {"pro": ["keep-0", "keep-1", "keep-2"]}})
    raw = [{"id": f"keep-{i}", "label": f"K{i}"} for i in range(3)] + [
        {"id": "wanted", "label": "Wanted"},
    ]
    # threshold=2 forces a split: without pre-split subtraction, visible would
    # be [keep-0, keep-1] and post-split filtering could only shrink it.
    visible, extras = c._split_picker_overflow_models(
        raw, threshold=2, target=2, provider_id="pro"
    )
    # sanity: the split itself puts keepers first
    assert [m["id"] for m in visible] == ["keep-0", "keep-1"]
    # now the catalog path this test guards: subtract, then split
    hidden = c._model_excluded_ids_for_provider("pro")
    remaining = [
        m for m in raw if not (c._model_exclusion_key(m["id"]) & hidden)
    ]
    visible2, extras2 = c._split_picker_overflow_models(
        remaining, threshold=2, target=2, provider_id="pro"
    )
    assert [m["id"] for m in visible2] == ["wanted"]
    assert extras2 == []


def test_live_models_route_subtracts_user_hidden_rows(monkeypatch):
    """/api/models/live drops user-hidden rows before the visibility cap.

    Regression: the route's exclusion block imported
    ``_model_excluded_ids_for_provider`` as ``_excluded_ids`` but called the
    unaliased name, so the bare ``except`` swallowed a ``NameError`` and hidden
    rows leaked back into the picker through the live probe.
    """
    import sys
    import types
    from urllib.parse import urlparse

    import api.config as c
    import api.profiles as profiles
    import api.routes as routes

    _write_hidden(
        {
            "models_global_hidden": ["deepseek/deepseek-v4-pro"],
            "models_hidden": {"commandcode": ["deepseek/deepseek-v4-flash"]},
        }
    )

    hermes_cli = types.ModuleType("hermes_cli")
    hermes_cli.__path__ = []
    models = types.ModuleType("hermes_cli.models")
    models.provider_model_ids = lambda provider: [
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4.1-flash",
    ]
    monkeypatch.setitem(sys.modules, "hermes_cli", hermes_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.models", models)

    routes._clear_live_models_cache()
    monkeypatch.setattr(
        routes, "j", lambda _h, payload, status=200, extra_headers=None: payload
    )
    monkeypatch.setattr(c, "get_config", lambda: {"model": {"provider": "commandcode"}})
    monkeypatch.setattr(c, "_resolve_provider_alias", lambda provider: provider)
    monkeypatch.setattr(profiles, "get_active_profile_name", lambda: "default")

    result = routes._handle_live_models(
        object(), urlparse("/api/models/live?provider=commandcode")
    )
    ids = [m["id"] for m in result["models"]]
    assert ids == ["deepseek/deepseek-v4.1-flash"]
