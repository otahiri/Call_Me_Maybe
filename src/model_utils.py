"""model utils"""

from llm_sdk import Small_LLM_Model
import json
from pydantic import BaseModel, ConfigDict, model_validator
from typing import Any


def set_module(model_sign: str) -> Small_LLM_Model:
    """set the model to the chosen one

    Args:
        module_sign: the sign of the model

    Returns:
        return the model generated

    Raises:
        ValueError: raised when the model_sign is not a valid sign
    """
    match model_sign:
        case "base":
            return Small_LLM_Model()
        case "coder":
            return Small_LLM_Model(
                    model_name="Qwen/Qwen2.5-Coder-0.5B"
                    )
        case _:
            raise ValueError("invalid model sign")


class ModelInterface(BaseModel):
    """model interface to make it easier to talk to the model

    Attributes:
        vocab: the vocab dictionary
        merge: the merge list of all the valid merges
        model: the small llm model
        special_chars: characters to replace to make the model
        understand better
        model_config: allow pydantic to accept custom types
        translation: vocab to reversed value: key for decoding
    """
    vocab: dict = dict()
    merge: dict = dict()
    model: Small_LLM_Model = set_module("base")
    special_chars: dict = {" ": "Ġ", "\n": "Ċ", "\t": "ĉ"}
    model_config = ConfigDict(arbitrary_types_allowed=True)
    translation: dict = dict()

    @model_validator(mode='after')
    def initiate_utils(self) -> Any:
        """initiate and validate values

        Returns:
            return the validated object

        Raises:
            ValueError: raised if an invalid file or invalid value detected
        """
        try:
            with open(self.model.get_path_to_vocab_file(), "r") as f:
                self.vocab = dict(json.load(f))
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            raise ValueError("could not access vocab file")
        try:
            with open(self.model.get_path_to_merges_file(), "r") as f:
                lines = f.read().splitlines()
                for rank, line in enumerate(lines):
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split(" ")
                    if len(parts) != 2:
                        raise ValueError("invalid merge file")
                    p1, p2 = parts
                    merge_str = "".join([p1, p2])
                    if (p1 in self.vocab and p2 in self.vocab
                            and merge_str in self.vocab):
                        id1, id2 = self.vocab[p1], self.vocab[p2]
                        self.merge[(id1, id2)] = (self.vocab[merge_str], rank)
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            raise ValueError("could not access merge file")
        self.translation: dict = {v: k for k, v in self.vocab.items()}
        return self

    def encode(self, prompt: str) -> list[int]:
        """convert prompt into numerical values and merge
        the appropriate ids using byte_pair merge algo

        Args:
            prompt: prompt to feed the model

        Returns:
            list of the encoded values
        """
        for key, value in self.special_chars.items():
            prompt = prompt.replace(key, value)
        tokens = [self.vocab.get(c, 0) for c in prompt]
        while True:
            pairs = [(tokens[i], tokens[i + 1])
                     for i in range(len(tokens) - 1)]
            pair_to_merge = None
            lowest_rank = float('inf')
            merge_id = None

            for p in pairs:
                if p in self.merge:
                    target_id, rank = self.merge[p]
                    if rank < lowest_rank:
                        lowest_rank = rank
                        merge_id = target_id
                        pair_to_merge = p
            if not pair_to_merge:
                break
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) == pair_to_merge:
                    tokens[i:i + 2] = [merge_id]
                else:
                    i += 1
        return tokens

    def decode(self, tokens:  list[int]) -> str:
        """convert tokens into a string

        Args:
            tokens: list of encoded values

        Returns:
            string generated from converting tokens
        """
        out = ""
        for t in tokens:
            out += self.translation.get(t, "")
        for key, value in self.special_chars.items():
            out = out.replace(value, key)
        return out
