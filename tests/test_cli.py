from __future__ import annotations

from ocular_evals.cli import main


def test_normalize_command(capsys):
    assert main(["normalize", "--language", "en", "I'm <breath/> here, okay?"]) == 0
    assert capsys.readouterr().out.strip() == "i am here okay"


def test_compare_command_with_inline_text(capsys, tmp_path):
    reference = tmp_path / "ref.txt"
    reference.write_text("Hello there friend.", encoding="utf-8")
    assert main(["compare", str(reference), "hello their friend", "--show"]) == 0
    out = capsys.readouterr().out
    assert "WER 33.33%" in out and "CER" in out and "hypothesis: hello their friend" in out


def test_models_command_lists_builtins(capsys):
    assert main(["models"]) == 0
    out = capsys.readouterr().out
    assert "omi-general-1" in out and "whisper-large-v3" in out


def test_models_add_openrouter_route(capsys, tmp_path, monkeypatch):
    registry = tmp_path / "models.toml"
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    code = main(["--models-file", str(registry), "models", "add", "acme-stt", "--openrouter", "--model", "acme/stt-1"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "ready (openai-compatible:acme/stt-1)" in out
    text = registry.read_text(encoding="utf-8")
    assert 'url = "https://openrouter.ai/api/v1"' in text and 'api_key_env = "OPENROUTER_API_KEY"' not in text
    assert 'key = "OPENROUTER_API_KEY"' in text and 'upload = "mp3"' in text and "via OpenRouter" in text
    # Direct entries still work next to routed ones, under their own name.
    assert (
        main(
            [
                "--models-file",
                str(registry),
                "models",
                "add",
                "acme-stt-direct",
                "--url",
                "https://api.acme.example/v1",
                "--key",
                "ACME_API_KEY",
                "--model",
                "stt-1",
            ]
        )
        == 1
    )  # key not set -> "not ready", but registered
    assert "acme-stt-direct" in registry.read_text(encoding="utf-8")
    # --openrouter needs a vendor/model id and refuses a conflicting --url.
    assert main(["--models-file", str(registry), "models", "add", "x", "--openrouter"]) == 2
    assert (
        main(
            [
                "--models-file",
                str(registry),
                "models",
                "add",
                "x",
                "--openrouter",
                "--model",
                "a/b",
                "--url",
                "https://y",
            ]
        )
        == 2
    )


def test_unknown_model_is_a_clean_error(capsys, tmp_path):
    assert main(["run", "--model", "nope", "--dataset", str(tmp_path)]) == 2
    assert "unknown model" in capsys.readouterr().err
