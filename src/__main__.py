import sys
import gc

from numpy import extract

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
        functions = [f['name'] for f in raw_funcs.functions]

        func_list = [(f['name'], f['description'])
                     for f in raw_funcs.functions]
        example_prompt = model.encode(
                    "You are a strict data extraction script."
                    " Your ONLY job is to identify the core action or function"
                    " requested in the user's input and output the "
                    "function name in snake_case followed immediately"
                    " by a space and the word \"end\".\n"
                    f"{func_list}\n"
                    f"each function name is paired with its description\n"
                    "EXAMPLES:\n"
                    "question: greet jake\n"
                    "answer: fn_greet end\n"
                    "question: add 2 and 9\n"
                    "answer: fn_add_numbers\n"
                    )

        allowed_token = []
        for name in functions:
            allowed_token.extend(model.encode(name))
        allowed_token.extend(model.encode(" end\n"))
        allowed_token = list(set(allowed_token))
        func_lookup = {f['name']: f for f in raw_funcs.functions}
        print("start")
        for p in prompts:
            prompt = (
                    f"question: {p}\n"
                    "answer"
                    )
            encoded = model.encode(prompt)
            encoded = example_prompt + encoded
            out = ""
            for _ in range(100):
                if out:
                    functions = [f for f in functions if out in f]
                if len(functions) == 1:
                    out = functions[0]
                    funcs.append(
                            Func(
                                name=out,
                                description=func_lookup[out]['description'],
                                prompt=p
                                )
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
                if "end" in out:
                    out = out.replace("end", "").strip()
                    funcs.append(
                            Func(name=out,
                                 description=func_lookup[out]['description'],
                                 prompt=p
                                 )
                                 )
                    break
                encoded.append(new_id)
            print(out)
        model.del_model()
        gc.collect()
        model.set_module("coder")

        example_prompt = model.encode(
            "CRITICAL: DO NOT generate code, DO NOT solve math problems, DO NOT apply regex rules\n"
            "if a prompt says replace x with y the replacement is y not the modified text, stick to the rules and do not apply your own logic\n"
            "You are a strict data extraction script. DO NOT solve the problems.\n"
            "do not edit the values from the prompt do not do anything to them\n"
            "you do NOT solve anything do NOT change anything just extract the parameters exactly as mentioned in the prompt\n"
            "Your ONLY job is to extract the exact value for the requested parameter from the prompt nothing more\n"
            "You must output ONLY the value, the exact value, surrounded by double quotes.\n\n"
            "EXAMPLES:\n"
            "name: fn_add_numbers\n"
            "prompt: What is the sum of 8 and 7?\n"
            "extracted parameter \"a\": \"8\"\n\n"
            "name: fn_add_numbers\n"
            "prompt: What is the sum of 8 and 7?\n"
            "extracted parameter \"b\": \"7\"\n\n"
            "name: fn_substitute_string_with_regex\n"
            "prompt: replace all M H Y and A with dash in this sentence \"Hello There MY NAme is ME\"\n"
            "extracted parameter \"source_string\": \"Hello There MY NAme is ME\"\n\n"
            "name: fn_substitute_string_with_regex\n"
            "prompt: replace all M H Y and A with dash in this sentence \"Hello There MY NAme is ME\"\n"
            "extracted parameter \"regex\": \"[MHYA]\"\n\n"
            "name: fn_substitute_string_with_regex\n"
            "prompt: replace all M H Y and A with dash in this sentence \"Hello There MY NAme is ME\"\n"
            "extracted parameter \"replacement\": \"-\"\n\n"
            "name: fn_substitute_string_with_regex\n"
            "prompt: replace all numbers with asterisks in 'hello my name is me and i432m happy892 to be 234 years'\n"
            "extracted parameter \"source_string\": \"hello my name is me and i432m happy892 to be 234 years\"\n\n"
            "name: fn_substitute_string_with_regex\n"
            "prompt: replace all numbers with asterisks in 'hello my name is me and i432m happy892 to be 234 years'\n"
            "extracted parameter \"regex\": \"[0-9]\"\n\n"
            "name: fn_substitute_string_with_regex\n"
            "prompt: replace all numbers with asterisks in 'hello my name is me and i432m happy892 to be 234 years'\n"
            "extracted parameter \"replacement\": \"*\"\n\n"
        )
        for f in funcs:
            prompt = (
                f"name: {f.name}\n"
                f"prompt: {f.prompt}\n"
                )
            f.params = {}

            base_encoded = example_prompt + model.encode(prompt)
            for param in func_lookup[f.name]["parameters"]:

                current_encoded = list(base_encoded)
                prefix_prompt = f'extracted parameters "{param}": "'
                current_encoded.extend(model.encode(prefix_prompt))
                out = ""
                for _ in range(200):
                    logits = model.model.get_logits_from_input_ids(current_encoded)
                    new_id = int(np.argmax(logits))
                    out += model.decode([new_id])
                    current_encoded.append(new_id)
                    print(out)
                    if '"' in out:
                        print(out)
                        break

                clean_val = out[:out.find('"')]
                f.params[param] = clean_val
                print(f)
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
