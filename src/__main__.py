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
                    "You are a strict data extraction script.\n"
                    " Your ONLY job is to identify the core action or function\n"
                    " requested in the user's input and output the \n"
                    "function name in snake_case followed immediately\n"
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
                "You are a strict data extraction script. DO NOT solve the problems.\n"
                "you are really good at making regex when asked for them\n"
                "Your only job is to extract the parameters for a function from a prompt\n"
                "DO NOT do anything with those parameters do not do any equation with them or use them for anything or do anything to them\n"
                "only give me the parameters surrounded by \"\n"
                "EXAMPLES:\n\n"
                "function name: fn_add_numbers\n"
                'expected parameters: "a": "number", "b": number\n'
                "prompt: add 2 and 3\n"
                'parameters: "a": "2", "b": "3"\n\n'
                "function name: fn_get_square_root\n"
                'expected parameters: "a": "number"\n'
                "prompt: what is the square root of 400\n"
                'parameters: "a": "400"\n\n'
                "function name: fn_substitute_string_with_regex\n"
                'expected parameters: "source_string": "string", "regex": "string" ,"replacement": "string"\n'
                'prompt: replace all digits in this string "hello 32 i am 432 years old i love the  number 32" with asterisk\n'
                'parameters: "source_string": "hello 32 i am 432 years old i love the  number 32", "regex": "[0-9]+" ,"replacement": "*"\n'
                )
        for f in funcs:
            expected_param = " ,".join([f'"{k}": "{v["type"]}"' for k, v in func_lookup[f.name]['parameters'].items()])
            prompt = (
                f"function name: {f.name}\n"
                f"expected parameters: {expected_param}"
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
