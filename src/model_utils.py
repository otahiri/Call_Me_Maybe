from llm_sdk import Small_LLM_Model
import json
import torch


class ModelInterface:
    def __init__(self) -> None:
        self.vocab: dict = dict()
        self.merge: dict = dict()
        self.model: Small_LLM_Model = Small_LLM_Model()
        self.special_chars: dict = {" ": "Ġ", "\n": "Ċ", "\t": "ĉ"}

        with open(self.model.get_path_to_vocab_file(), "r") as f:
            self.vocab = dict(json.load(f))
        # with open(self.model.get_path_to_tokenizer_file, 'r') as f:
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
                    self.merge[(p1, p2)] = (self.vocab[merge_str], rank)
        self.translation: dict = {v: k for k, v in self.vocab.items()}

    def encode(self, prompt: str) -> torch.Tensor:
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
            for i in range(len(tokens) - 1):
                if (tokens[i], tokens[i + 1]) == pair_to_merge:
                    tokens[i:i + 2] = [merge_id]
        return torch.tensor([tokens], device=self.model._device, dtype=torch.long)

    def decode(self, tokens: torch.Tensor | list[int]) -> str:
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        out = self.translation.get(tokens, "")
        for key, value in self.special_chars.items():
            out = out.replace(value, key)
        return out
