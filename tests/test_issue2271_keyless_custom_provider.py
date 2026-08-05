from __future__ import annotations


def test_keyless_named_custom_provider_uses_placeholder_and_generic_custom(monkeypatch):
    import api.streaming as streaming

    monkeypatch.setattr(
        streaming,
        "resolve_custom_provider_connection",
        lambda provider: (None, "http://gpu.local:8000/v1"),
    )

    provider, api_key, base_url = streaming._resolve_custom_provider_runtime_overrides(
        "custom:gpu-local-8000", None, None
    )

    assert provider == "custom"
    assert api_key == "dummy-key"
    assert base_url == "http://gpu.local:8000/v1"


def test_named_custom_provider_preserves_configured_key(monkeypatch):
    import api.streaming as streaming

    monkeypatch.setattr(
        streaming,
        "resolve_custom_provider_connection",
        lambda provider: ("real-key", "http://gpu.local:8000/v1"),
    )

    provider, api_key, base_url = streaming._resolve_custom_provider_runtime_overrides(
        "custom:gpu-local-8000", None, None
    )

    # A real api key means the account is NOT keyless: keep the named
    # custom:<slug> identity so Agent init routes to this exact account
    # instead of the first account sharing the base URL.
    assert provider == "custom:gpu-local-8000"
    assert api_key == "real-key"
    assert base_url == "http://gpu.local:8000/v1"


def test_named_custom_provider_with_explicit_key_keeps_identity(monkeypatch):
    import api.streaming as streaming

    monkeypatch.setattr(
        streaming,
        "resolve_custom_provider_connection",
        lambda provider: (None, "http://gpu.local:8000/v1"),
    )

    # Caller already resolved a key (explicit_api_key path): keep the named
    # identity as well — a keyed account must never collapse to bare custom.
    provider, api_key, base_url = streaming._resolve_custom_provider_runtime_overrides(
        "custom:gpu-local-8000", "caller-key", None
    )

    assert provider == "custom:gpu-local-8000"
    assert api_key == "caller-key"
    assert base_url == "http://gpu.local:8000/v1"


def test_named_custom_provider_keeps_existing_runtime_base_url(monkeypatch):
    import api.streaming as streaming

    monkeypatch.setattr(
        streaming,
        "resolve_custom_provider_connection",
        lambda provider: (None, "http://config.example/v1"),
    )

    provider, api_key, base_url = streaming._resolve_custom_provider_runtime_overrides(
        "custom:runtime-local", None, "http://runtime.example/v1"
    )

    assert provider == "custom"
    assert api_key == "dummy-key"
    assert base_url == "http://runtime.example/v1"


def test_non_custom_provider_is_unchanged(monkeypatch):
    import api.streaming as streaming

    called = False

    def _unexpected(provider):
        nonlocal called
        called = True
        return (None, None)

    monkeypatch.setattr(streaming, "resolve_custom_provider_connection", _unexpected)

    provider, api_key, base_url = streaming._resolve_custom_provider_runtime_overrides(
        "openrouter", None, None
    )

    assert (provider, api_key, base_url) == ("openrouter", None, None)
    assert called is False


def test_custom_provider_env_name_is_posix_safe():
    import api.config as config

    assert config._api_key_env_name("custom:gpu.local-8000") == "CUSTOM_GPU_LOCAL_8000_API_KEY"
    assert config._api_key_env_name("custom:10.8.71.41:8080") == "CUSTOM_10_8_71_41_8080_API_KEY"
    assert config._api_key_env_name("custom/foo bar") == "CUSTOM_FOO_BAR_API_KEY"


def test_resolve_custom_provider_connection_prefers_sanitized_env(monkeypatch):
    import api.config as config

    monkeypatch.setattr(
        config,
        "get_config",
        lambda: {
            "custom_providers": [
                {"name": "gpu.local-8000", "base_url": "http://gpu.local:8000/v1"},
            ],
        },
    )
    monkeypatch.setenv("CUSTOM_GPU_LOCAL_8000_API_KEY", "sanitized-key")
    monkeypatch.setenv("CUSTOM:GPU.LOCAL-8000_API_KEY", "legacy-key")

    api_key, base_url = config.resolve_custom_provider_connection("custom:gpu.local-8000")

    assert api_key == "sanitized-key"
    assert base_url == "http://gpu.local:8000/v1"


def test_resolve_custom_provider_connection_falls_back_to_legacy_env(monkeypatch, caplog):
    import logging
    import api.config as config

    config._LEGACY_CUSTOM_API_KEY_ENV_WARNED.clear()
    monkeypatch.setattr(
        config,
        "get_config",
        lambda: {
            "custom_providers": [
                {"name": "gpu.local-8000", "base_url": "http://gpu.local:8000/v1"},
            ],
        },
    )
    monkeypatch.delenv("CUSTOM_GPU_LOCAL_8000_API_KEY", raising=False)
    monkeypatch.setenv("CUSTOM:GPU.LOCAL-8000_API_KEY", "legacy-key")

    with caplog.at_level(logging.WARNING, logger="api.config"):
        api_key, _base_url = config.resolve_custom_provider_connection("custom:gpu.local-8000")

    assert api_key == "legacy-key"
    assert "CUSTOM_GPU_LOCAL_8000_API_KEY" in caplog.text


def test_resolve_custom_provider_connection_keyed_providers_unprefixed_slug(monkeypatch):
    """Keyed ``providers:`` dict stores the account under its bare slug, not
    the ``custom:<slug>`` prefixed id; the resolver must find it either way."""
    import api.config as config

    monkeypatch.setattr(
        config,
        "get_config",
        lambda: {
            "providers": {
                "vercel-vtest314": {
                    "api": "https://ai-gateway.vercel.sh/v1",
                    "api_key": "vtest-key",
                },
            },
        },
    )

    api_key, base_url = config.resolve_custom_provider_connection("custom:vercel-vtest314")

    assert api_key == "vtest-key"
    assert base_url == "https://ai-gateway.vercel.sh/v1"


def test_resolve_custom_provider_connection_keyed_providers_api_field(monkeypatch):
    """Keyed entries spell the endpoint as ``api`` (not legacy ``base_url``)."""
    import api.config as config

    monkeypatch.setattr(
        config,
        "get_config",
        lambda: {
            "providers": {
                "vercel-playground": {
                    "api": "https://ai-gateway.vercel.sh/v1",
                    "key_env": "VERCEL_PLAYGROUND_API_KEY",
                },
            },
        },
    )
    monkeypatch.setenv("VERCEL_PLAYGROUND_API_KEY", "play-key")

    api_key, base_url = config.resolve_custom_provider_connection("custom:vercel-playground")

    assert api_key == "play-key"
    assert base_url == "https://ai-gateway.vercel.sh/v1"


def test_resolve_custom_provider_connection_keyed_providers_no_match(monkeypatch):
    """No matching keyed entry -> (None, None), not a crash."""
    import api.config as config

    monkeypatch.setattr(
        config,
        "get_config",
        lambda: {
            "providers": {
                "vercel-vtest314": {"api": "https://ai-gateway.vercel.sh/v1"},
            },
        },
    )

    api_key, base_url = config.resolve_custom_provider_connection("custom:does-not-exist")

    assert (api_key, base_url) == (None, None)
