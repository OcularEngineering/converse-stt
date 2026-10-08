from __future__ import annotations

import pytest

from ocular_evals.config import ConfigError, ModelConfig, builtin_model_names, load_model_config
from ocular_evals.providers import PROVIDERS, create_provider

EXPECTED = {
    "gpt-4o-mini-transcribe",
    "gpt-transcribe",
    "omi-general-1",
    "omi-medical-1",
    "whisper-large-v3",
    "whisper-tiny",
}


def test_builtin_model_names_include_the_shipped_configs():
    assert EXPECTED <= set(builtin_model_names())


@pytest.mark.parametrize("name", builtin_model_names())
def test_builtin_config_loads_and_provider_checks(name: str):
    cfg = load_model_config(name)
    assert cfg.name == name
    assert cfg.provider in PROVIDERS
    provider = create_provider(cfg)
    assert provider.key == cfg.provider
    problems = provider.check()
    assert isinstance(problems, list)
    assert all(isinstance(item, str) for item in problems)
    # Only environment / dependency problems are acceptable for shipped configs; never unknown options.
    assert not any("unknown option" in item for item in problems), problems


def test_unknown_provider_raises_config_error():
    cfg = ModelConfig(name="x", display_name="x", vendor="x", provider="nope", model="m")
    with pytest.raises(ConfigError, match="unknown provider 'nope'"):
        create_provider(cfg)
