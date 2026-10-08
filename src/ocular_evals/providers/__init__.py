"""Speech-to-text providers keyed by the ``provider`` field of a model config."""

from __future__ import annotations

from ocular_evals.config import ConfigError, ModelConfig
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError
from ocular_evals.providers.faster_whisper import FasterWhisperProvider
from ocular_evals.providers.omi import OmiProvider
from ocular_evals.providers.openai_compatible import OpenAICompatibleProvider

PROVIDERS: dict[str, type[Provider]] = {
    OpenAICompatibleProvider.key: OpenAICompatibleProvider,
    OmiProvider.key: OmiProvider,
    FasterWhisperProvider.key: FasterWhisperProvider,
}


def create_provider(config: ModelConfig) -> Provider:
    try:
        cls = PROVIDERS[config.provider]
    except KeyError:
        available = ", ".join(sorted(PROVIDERS))
        raise ConfigError(f"unknown provider {config.provider!r}; available: {available}") from None
    return cls(config)


__all__ = ["Hypothesis", "Provider", "ProviderError", "PROVIDERS", "create_provider"]
