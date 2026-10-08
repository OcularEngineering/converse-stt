from __future__ import annotations

import os

import pytest

from ocular_evals.cli import main
from ocular_evals.config import ConfigError, all_model_names, load_model_config
from ocular_evals.env import parse_dotenv
from ocular_evals.registry import Registry, normalize_url, suggest_env_name, upsert_dotenv

MINIMAL = """
# my models
[deepgram-nova-3]
url = "https://api.deepgram.com/v1/audio/transcriptions/"
key = "DEEPGRAM_API_KEY"

[acme-stt]
url = "https://api.acme.ai/v1"
key = "ACME_API_KEY"
model = "acme-stt-2"
vendor = "Acme"
display_name = "Acme STT 2"
upload = "mp3"
"""


def test_two_line_entry_expands_to_openai_compatible_config(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(MINIMAL, encoding="utf-8")
    registry = Registry.load(path)
    assert registry.names() == ["deepgram-nova-3", "acme-stt"]

    config = registry.config("deepgram-nova-3")
    assert config.provider == "openai-compatible"
    assert config.model == "deepgram-nova-3"
    assert config.env == ("DEEPGRAM_API_KEY",)
    assert config.options == {"base_url": "https://api.deepgram.com/v1", "api_key_env": "DEEPGRAM_API_KEY"}

    acme = registry.config("acme-stt")
    assert acme.model == "acme-stt-2" and acme.vendor == "Acme" and acme.display_name == "Acme STT 2"
    assert acme.options["upload"] == "mp3"


def test_registry_shadows_builtins_and_lists_first(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text('[omi-general-1]\nkey = "OMI_API_KEY"\nprovider = "omi"\nvendor = "Omi"\n', encoding="utf-8")
    config = load_model_config("omi-general-1", registry=path)
    assert config.source.endswith(":[omi-general-1]")
    names = all_model_names(registry=path)
    assert names[0] == "omi-general-1" and names.count("omi-general-1") == 1
    assert "whisper-tiny" in names


def test_secret_in_key_field_is_rejected(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text('[leaky]\nurl = "https://x.ai/v1"\nkey = "sk-live-abcdef"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="NAME of an environment variable"):
        Registry.load(path).config("leaky")


def test_transport_options_do_not_change_the_digest(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(
        '[a]\nurl = "https://one.example/v1"\nkey = "A_KEY"\n\n'
        '[b]\nurl = "https://two.example/v1"\nkey = "B_KEY"\nmodel = "a"\n\n'
        '[c]\nurl = "https://one.example/v1"\nkey = "A_KEY"\nmodel = "a"\nupload = "mp3"\n',
        encoding="utf-8",
    )
    registry = Registry.load(path)
    assert registry.config("a").digest() == registry.config("b").digest()
    assert registry.config("a").digest() != registry.config("c").digest()


def test_add_appends_and_keeps_comments_remove_rewrites(tmp_path):
    path = tmp_path / "models.toml"
    path.write_text(MINIMAL, encoding="utf-8")
    registry = Registry.load(path)
    registry.add("new-one", url="https://api.new.example/v1/", key_env="NEW_API_KEY", vendor="New")
    registry.save()
    text = path.read_text(encoding="utf-8")
    assert "# my models" in text and text.rstrip().endswith('vendor = "New"')
    assert Registry.load(path).names() == ["deepgram-nova-3", "acme-stt", "new-one"]

    with pytest.raises(ConfigError, match="already exists"):
        registry.add("new-one", url="https://api.new.example/v1", key_env="NEW_API_KEY")

    registry.remove("acme-stt")
    registry.save()
    reloaded = Registry.load(path)
    assert reloaded.names() == ["deepgram-nova-3", "new-one"]
    assert reloaded.config("new-one").options["base_url"] == "https://api.new.example/v1"


def test_helpers():
    assert normalize_url("https://api.x.com/v1/audio/transcriptions") == "https://api.x.com/v1"
    assert normalize_url("https://api.x.com/v1/") == "https://api.x.com/v1"
    assert suggest_env_name("deepgram-nova-3") == "DEEPGRAM_API_KEY"
    assert suggest_env_name("gpt-4o") == "GPT_API_KEY"


def test_upsert_dotenv_replaces_or_appends(tmp_path):
    env = tmp_path / ".env"
    env.write_text("A=1\nB=2\n", encoding="utf-8")
    upsert_dotenv(env, "B", "two")
    upsert_dotenv(env, "C", "th ree")
    assert parse_dotenv(env.read_text(encoding="utf-8")) == {"A": "1", "B": "two", "C": "th ree"}


def test_cli_models_add_with_literal_key_writes_dotenv(tmp_path, capsys, monkeypatch):
    registry = tmp_path / "models.toml"
    env = tmp_path / ".env"
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    code = main(
        [
            "--models-file",
            str(registry),
            "--env-file",
            str(env),
            "models",
            "add",
            "deepgram-nova-3",
            "--url",
            "https://api.deepgram.com/v1",
            "--key",
            "dg_secret_value_1234",
            "--model",
            "nova-3",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "ready" in out and "dg_secret_value_1234" not in out
    assert parse_dotenv(env.read_text(encoding="utf-8")) == {"DEEPGRAM_API_KEY": "dg_secret_value_1234"}
    assert "dg_secret_value_1234" not in registry.read_text(encoding="utf-8")
    assert os.environ["DEEPGRAM_API_KEY"] == "dg_secret_value_1234"

    assert main(["--models-file", str(registry), "models", "show", "deepgram-nova-3"]) == 0
    assert '"model": "nova-3"' in capsys.readouterr().out

    # Default `check` covers every registered model; --model works too.
    assert main(["--models-file", str(registry), "--env-file", str(env), "check"]) == 0
    assert main(["--models-file", str(registry), "--env-file", str(env), "check", "--model", "deepgram-nova-3"]) == 0


def test_cli_models_add_requires_url_and_key(tmp_path, capsys):
    registry = tmp_path / "models.toml"
    assert main(["--models-file", str(registry), "models", "add", "x", "--key", "X_API_KEY"]) == 2
    assert "--url is required" in capsys.readouterr().err
    assert main(["--models-file", str(registry), "models", "add", "x", "--url", "https://x/v1"]) == 2
    assert "--key is required" in capsys.readouterr().err
    assert not registry.exists()
