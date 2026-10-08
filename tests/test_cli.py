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


def test_unknown_model_is_a_clean_error(capsys, tmp_path):
    assert main(["run", "--model", "nope", "--dataset", str(tmp_path)]) == 2
    assert "unknown model" in capsys.readouterr().err
