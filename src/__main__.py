from os import replace
import sys

from pydantic import EncoderProtocol

try:
    from typing import Any
    import argparse

    import pydantic
    from src import RawInputs, ModelInterface, Func, Param

    import json
    import numpy as np
    import torch

    # def test(prompt: str):
    #     prompt = (
    #             "You are a strict data extraction script. Your ONLY job is to identify the core action or function requested in the user's input and output the function name in snake_case followed immediately by a space and the word \"end\"."
    #             ""
    #             "RULES:"
    #             "1. Output exactly the function name and the word \"end\"."
    #             "2. Do NOT output any other text, no explanations, no punctuation, no \"Sure\", and no greetings."
    #             "3. If the action is unclear, output: unknown_action end"
    #             ""
    #             "EXAMPLES:"
    #             "Input: \"can you turn off the living room lights for me?\""
    #             "Output: turn_off_lights end"
    #             ""
    #             "Input: \"what's the weather like in Tokyo right now?\""
    #             "Output: get_weather end"
    #             ""
    #             "Input: \"create a new file called test.txt\""
    #             "Output: create_file end"
    #             ""
    #             "Input: \"fetch me the user details for ID 45\""
    #             "Output: get_user_details end"
    #             ""
    #             f"Input: \"{prompt}\""
    #             "Output:"
    # )
    #     encoded = model.encode(prompt)
    #
    #     max_token = 200
    #     answer = ""
    #
    #     for _ in range(max_token):
    #         logits = model.get_logits_from_input_ids(encoded.squeeze(0).tolist())
    #         new_id = np.argmax(np.array(logits))
    #         token_text = model.decode(torch.tensor(new_id))
    #         answer += token_text
    #         print(token_text, end='')
    #         if "end" in answer:
    #             break
    #         new_id_tensor = torch.tensor([[new_id]], device=encoded.device)
    #         encoded = torch.cat((encoded, new_id_tensor), dim=1)
    #     print()

    # "name": "fn_add_numbers",
    # "description": "Add two numbers together and return their sum.",
    # "parameters": {
    #   "a": {
    #     "type": "number"
    #   },
    #   "b": {
    #     "type": "number"
    #   }
    # },
    # "returns": {
    #   "type": "number"
    # }

    def main() -> None:
        prompts = []
        data: Any
        funcs: Any
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--functions_definition",
            type=str,
            default="data/input/functions_definition.json",
        )

        parser.add_argument(
            "--input", type=str,
            default="data/input/function_calling_tests.json"
        )
        parser.add_argument(
                "--output", type=str, default="data/output/function_calls.json"
                )
        # with open(model.get_path_to_vocab_file(), "r") as f:
        #     vocab = json.load(f)
        args = parser.parse_args()
        with open(args.functions_definition, "r") as func_file:
            funcs = json.load(func_file)

        with open(args.input, "r") as file:
            data = json.load(file)
            for line in data:
                for key, value in line.items():
                    if key == "prompt":
                        prompts.append(value)
                    else:
                        raise ValueError(
                                "error prompts should be {prompt: prompt}"
                                )

        raw_funcs = RawInputs(functions=funcs, prompts=prompts)
        funcs = [Func(name=f['name'], description=f['description'],
                      params={k: Param(param_type=v['type'])
                      for k, v in f['parameters'].items()})
                 for f in raw_funcs.functions]
        for f in funcs:
            print(f)
        model = ModelInterface()
        for p in prompts:
            prompt = (
                    "You are a strict data extraction script."
                    " Your ONLY job is to identify the core action or function"
                    " requested in the user's input and output the "
                    "function name in snake_case followed immediately"
                    " by a space and the word \"end\".\n"
                    "the allowed functions are:\n"
                    f"{[(f.name, f.description) for f in funcs]}\n"
                    f"each function name is paired with its description\n"
                    "EXAMPLES:\n"
                    "question: greet jake\n"
                    "answer: fn_greet end\n"
                    "question: add 2 and 9\n"
                    "answer: fn_add_numbers\n"
                    f"question: {p}\n"
                    "answer"
                    )
            encoded = model.encode(prompt)
            out = ""
            functions = [f.name for f in funcs]
            allowed_token = []
            for name in functions:
                allowed_token.extend(model.encode(name).flatten().tolist())
            allowed_token.extend(model.encode(" end\n").flatten().tolist())
            allowed_token = list(set(allowed_token))
            for _ in range(100):
                if out:
                    functions = [f for f in functions
                                 if f.startswith(out)]
                if len(functions) == 1:
                    out = functions[0]
                    break
                # if len(functions) == 0:
                #     print("i guess it got fucked up somewhere")
                #     break
                logits = model.model.get_logits_from_input_ids(
                        encoded.squeeze(0).tolist()
                        )
                mask = np.full_like(logits, -np.inf)
                for token in allowed_token:
                    mask[token] = logits[token]
                logits = mask
                new_id = int(np.argmax(logits))
                text = model.decode([int(new_id)])
                out += text
                if " end" in out:
                    out = out.replace(" end", "")
                    break
                new_id_tensor = torch.tensor([[new_id]],
                                             device=model.model._device)
                encoded = torch.cat((encoded, new_id_tensor), dim=1)
            print(out)

    if __name__ == "__main__":
        try:
            main()
        except pydantic.ValidationError as e:
            print(e.errors()[0]["msg"], file=sys.stderr)
        except (IsADirectoryError, FileNotFoundError, PermissionError):
            print("invalid file", file=sys.stderr)
        except json.decoder.JSONDecodeError:
            print("invalid json file", file=sys.stderr)
except KeyboardInterrupt:
    print("user force stopped the program", file=sys.stderr)
