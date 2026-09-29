"""Provider-level ``extra_body`` must reach the main-model request overrides.

``providers.<name>.extra_body`` (for example the Vercel AI Gateway
``providerOptions.gateway.caching: auto`` switch that Anthropic models need) was
only honoured on the Agent CLI path.  The WebUI built its per-request overrides
from ``model.extra_body`` alone, so the setting never reached a WebUI turn.
"""

from api.config import _main_model_request_overrides

VERCEL = "https://ai-gateway.vercel.sh/v1"


def _cfg(**model):
    return {
        "model": model,
        "providers": {
            "vercel-playground": {
                "api": VERCEL,
                "extra_body": {"providerOptions": {"gateway": {"caching": "auto"}}},
            },
            "vercel-brandon-pro": {
                "api": VERCEL,
                "extra_body": {"providerOptions": {"gateway": {"caching": "auto", "sort": "tps"}}},
            },
            "plain": {"api": "https://example.invalid/v1"},
        },
    }


def test_provider_extra_body_reaches_overrides():
    out = _main_model_request_overrides(
        _cfg(default="anthropic/claude-sonnet-5.5"),
        effective_model="anthropic/claude-sonnet-5.5",
        effective_provider="vercel-playground",
    )
    assert out["extra_body"] == {"providerOptions": {"gateway": {"caching": "auto"}}}


def test_custom_prefixed_provider_name_resolves():
    out = _main_model_request_overrides(
        _cfg(default="x"),
        effective_model="anthropic/claude-sonnet-5.5",
        effective_provider="custom:vercel-playground",
    )
    assert out["extra_body"]["providerOptions"]["gateway"]["caching"] == "auto"


def test_only_the_selected_provider_is_used():
    out = _main_model_request_overrides(
        _cfg(default="x"),
        effective_model="m",
        effective_provider="vercel-brandon-pro",
    )
    assert out["extra_body"]["providerOptions"]["gateway"]["sort"] == "tps"
    other = _main_model_request_overrides(
        _cfg(default="x"), effective_model="m", effective_provider="vercel-playground"
    )
    assert "sort" not in other["extra_body"]["providerOptions"]["gateway"]


def test_provider_without_extra_body_adds_nothing():
    assert _main_model_request_overrides(
        _cfg(default="x"), effective_model="m", effective_provider="plain"
    ) == {}
    assert _main_model_request_overrides(
        _cfg(default="x"), effective_model="m", effective_provider="commandcode"
    ) == {}


def test_model_level_extra_body_wins_on_conflict_and_merges_the_rest():
    out = _main_model_request_overrides(
        _cfg(default="x", extra_body={"providerOptions": {"gateway": {"caching": "off"}}, "top": 1}),
        effective_model="m",
        effective_provider="vercel-playground",
    )
    assert out["extra_body"]["providerOptions"]["gateway"]["caching"] == "off"
    assert out["extra_body"]["top"] == 1


def test_provider_config_is_not_mutated():
    cfg = _cfg(default="x")
    out = _main_model_request_overrides(cfg, effective_model="m", effective_provider="vercel-playground")
    out["extra_body"]["providerOptions"]["gateway"]["caching"] = "mutated"
    assert cfg["providers"]["vercel-playground"]["extra_body"]["providerOptions"]["gateway"]["caching"] == "auto"


def test_non_dict_providers_block_is_ignored():
    assert _main_model_request_overrides(
        {"model": {"default": "x"}, "providers": ["oops"]}, effective_model="m", effective_provider="p"
    ) == {}
