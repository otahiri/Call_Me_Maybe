"""test cases to make sure the program does not crash"""

from pytest import MonkeyPatch
import sys
from src.__main__ import cli


class FakeModel:
    """dummy model to test"""
    def get_path_to_vocab_file(self) -> str:
        """dummy get_path_to_vocab_file for testing

        Returns:
            path to a dummy vocab file for testing
        """
        return "tests/fixtures/vocab.json"

    def get_path_to_merges_file(self) -> str:
        """dummy get_path_to_merges_file for testing

        Returns:
            path to a dummy merge file for testing
        """
        return "tests/fixtures/merges.txt"

    def get_logits_from_input_ids(self, ids: list[int]) -> None:
        """dummy get_logits_from_input_ids for testing

        Args:
            ids: list of tokens
        """
        print(f"mock get logits test with value {ids}")


def test_empty_func_def() -> None:
    """test empty function definition file"""
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--functions_definition", "tests/empty_func_test.json"])
    cli()


def test_no_func_def_file() -> None:
    """test no function definition file"""
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--functions_definition", "nofile"])
    cli()


def test_directory_error() -> None:
    """test functions definition directory"""
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--functions_definition",
             "tests/directory_error.json"])
    cli()


def test_parameter_error() -> None:
    """test invalid parameters"""
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--functions_definition", "tests/parameter_error.json"])
    cli()


def test_return_error() -> None:
    """test invalid return value"""
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--functions_definition", "tests/return_type_error.json"])
    cli()


def test_empty_prompt() -> None:
    """test empty prompt"""
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--input", "tests/empty_prompt_error.json"])
    cli()


def test_invalid_prompt_param() -> None:
    """test invalid prompt key"""
    monkeypatch = MonkeyPatch()
    monkeypatch.setattr("src.model_utils.Small_LLM_Model", FakeModel)
    monkeypatch.setattr(
            sys, "argv",
            ["src", "--input", "tests/prompt_key_error.json"])
    cli()


if __name__ == "__main__":
    try:
        test_empty_func_def()
        test_no_func_def_file()
        test_directory_error()
        test_parameter_error()
        test_return_error()
        test_empty_prompt()
        test_invalid_prompt_param()
    except Exception as e:
        print(e)
