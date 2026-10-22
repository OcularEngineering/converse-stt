"""Speech-to-text providers keyed by the ``provider`` field of a model config."""

from __future__ import annotations

from ocular_evals.config import ConfigError, ModelConfig
from ocular_evals.providers.base import Hypothesis, Provider, ProviderError
from ocular_evals.providers.cohere_transcribe import CohereTranscribeProvider
from ocular_evals.providers.faster_whisper import FasterWhisperProvider
from ocular_evals.providers.meta import MetaProvider
from ocular_evals.providers.nemotron_asr import NemotronAsrProvider
from ocular_evals.providers.omi import OmiProvider
from ocular_evals.providers.openai_compatible import OpenAICompatibleProvider
from ocular_evals.providers.openai_realtime import OpenAIRealtimeProvider
from ocular_evals.providers.reson8 import Reson8Provider
from ocular_evals.providers.xai import XaiProvider

PROVIDERS: dict[str, type[Provider]] = {
    OpenAICompatibleProvider.key: OpenAICompatibleProvider,
    OpenAIRealtimeProvider.key: OpenAIRealtimeProvider,
    XaiProvider.key: XaiProvider,
    MetaProvider.key: MetaProvider,
    Reson8Provider.key: Reson8Provider,
    OmiProvider.key: OmiProvider,
    FasterWhisperProvider.key: FasterWhisperProvider,
    NemotronAsrProvider.key: NemotronAsrProvider,
    CohereTranscribeProvider.key: CohereTranscribeProvider,
}


def create_provider(config: ModelConfig) -> Provider:
    try:
        cls = PROVIDERS[config.provider]
    except KeyError:
        available = ", ".join(sorted(PROVIDERS))
        raise ConfigError(f"unknown provider {config.provider!r}; available: {available}") from None
    return cls(config)


__all__ = ["Hypothesis", "Provider", "ProviderError", "PROVIDERS", "create_provider"]
