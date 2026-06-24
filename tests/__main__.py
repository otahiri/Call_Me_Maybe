from pytest import MonkeyPatch
import sys
from src.__main__ import cli


class FakeModel:
    def get_path_to_vocab_file(self): return "tests/fixtures/vocab.json"

    def get_path_to_merges_file(self): return "tests/fixtures/merges.txt"

    def get_logits_from_input_ids(self, ids):
        print(f"mock get logits test with value {ids}")


def test_empty_func_def() -> None:
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--functions_definition", "tests/empty_func_test.json"])
    cli()


def test_no_func_def_file() -> None:
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--functions_definition", "nofile"])
    cli()


def test_directory_error():
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--functions_definition",
             "tests/test_permission_error.json"])
    cli()


def test_parameter_error() -> None:
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--functions_definition", "tests/parameter_error.json"])
    cli()


def test_return_error() -> None:
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--functions_definition", "tests/return_type_error.json"])
    cli()


def test_empty_prompt() -> None:
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--input", "tests/empty_prompt_error.json"])
    cli()


def test_invalid_prompt_param() -> None:
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--input", "tests/prompt_key_error.json"])
    cli()


if __name__ == "__main__":
    test_empty_func_def()
    test_no_func_def_file()
    test_directory_error()
    test_parameter_error()
    test_return_error()
    test_empty_prompt()
    test_invalid_prompt_param()
