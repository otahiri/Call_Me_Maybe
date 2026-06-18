import sys
import gc

from numpy import mod

try:
    from typing import Any
    import argparse

    import pydantic
    from src import RawInputs, ModelInterface, Func

    import json
    import numpy as np

    def main() -> None:
        prompts = []
        model = ModelInterface()
        model.set_module("base")
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
        funcs = []
        for p in prompts:
            func_list = [
                    (f['name'], f['description']) for f in raw_funcs.functions
                    ]
            prompt = (
                    "You are a strict data extraction script."
                    " Your ONLY job is to identify the core action or function"
                    " requested in the user's input and output the "
                    "function name in snake_case followed immediately"
                    " by a space and the word \"end\".\n"
                    "the allowed functions are:\n"
                    f"{func_list}\n"
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
            functions = [f['name'] for f in raw_funcs.functions]
            allowed_token = []
            for name in functions:
                allowed_token.extend(model.encode(name))
            allowed_token.extend(model.encode(" end\n"))
            allowed_token = list(set(allowed_token))
            for _ in range(100):
                if out:
                    functions = [f for f in functions
                                 if f.startswith(out)]
                if len(functions) == 1:
                    print('here')
                    out = out.strip()
                    funcs.append(
                            Func(name=out,
                                 description=next(
                                     (f['description']
                                      for f in raw_funcs.functions
                                      if f['name'] == out),
                                     "invlaid function"
                                     ),
                                 prompt=p)
                            )
                    break
                logits = model.model.get_logits_from_input_ids(encoded)
                mask = np.full_like(logits, -np.inf)
                for token in allowed_token:
                    mask[token] = logits[token]
                logits = mask
                new_id = int(np.argmax(logits))
                text = model.decode([int(new_id)])
                out += text
                if " end" in out:
                    out = out.replace(" end", "").strip()
                    funcs.append(Func(name=out,
                                      description=next(
                                          (f['description']
                                           for f in raw_funcs.functions
                                           if f['name'] == out),
                                          "invlaid function"),
                                      prompt=p))
                    break
                encoded.append(new_id)
            print(out)
        model.del_model()
        gc.collect()
        model.set_module("coder")
        for f in funcs:
            correct_func = next((func for func in raw_funcs.functions if func['name'] == f.name))
            params = correct_func.get('parameters', {})
            out = "{"
            prompt = (
                "You are a strict data extraction script.\n"
                "Your ONLY job is to extract the parameters for a function from a prompt that will be provided, followed immediately by \" end\".\n\n"
                "you will not do anything with those parameters just provide the parameters followed immediately by \" end\n\""
                "EXAMPLES:\n"
                "name: fn_add_numbers\n"
                "description: Add two numbers together and return their sum.\n"
                "expected parameters: {'a': {'type': 'number'}, 'b': {'type': 'number'}}\n"
                "prompt: What is the sum of 8 and 7?\n"
                "extracted parameters: {\"a\": 8, \"b\": 7} end\n\n"
                "name: fn_substitute_string_with_regex\n"
                "description: Replace all occurrences matching a regex pattern in a string.\n"
                "expected parameters:\"source_string\": \"string\" , \"regex\": \"string\", \"replacement\":  \"string\"\n"
                "prompt: replace all M H Y and A with dash in this sentence \"Hello There, MY NAme is ME\"\n"
                "extracted parameters: {source_string: \"Hello There, MY NAme is ME\", regex: [MHYA], replacement: \"-\" } end\n"
                "name: fn_substitute_string_with_regex\n"
                "description: Replace all occurrences matching a regex pattern in a string.\n"
                "expected parameters:\"source_string\": \"string\" , \"regex\": \"string\", \"replacement\":  \"string\"\n"
                "prompt: replace all numbers with 0 in this sentence\"hello i have 3298 formats for all my friend i need 324 more\"\n"
                "extracted parameters: {source_string: \"hello i have 3298 formats for all my friend i need 324 more\", regex: [0-9], replacement: \"0\" } end\n"
                f"name: {f.name}\n"
                f"description: {f.description}\n"
                f"expected parameters: {params}\n"
                f"prompt: {f.prompt}\n"
                "extracted parameters: {"
            )

            encoded = model.encode(prompt)
            for _ in range(200):
                logits = model.model.get_logits_from_input_ids(encoded)
                new_id = int(np.argmax(logits))
                out += model.decode([new_id])
                encoded.append(new_id)

                if "end" in out:
                    print(out)
                    break
        model.del_model()
        del model
        gc.collect()

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

