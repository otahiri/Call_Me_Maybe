import sys
import gc

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

        func_list = [f['name'] for f in raw_funcs.functions]
        func_prompt = model.encode(
                f"function list = [{' ,'.join(func_list)}]\n"
                "prompt: add 2 and 3\n"
                '"answer": "fn_add_numbers"\n'
                'prompt: replace all number in this sentence "hello my name is me and 31 i 213 am 65 89 years old" with "-"\n'
                '"answer": "fn_substitute_string_with_regex"\n'
                "prompt: calculate the square root of 64\n"
                '"answer": "fn_get_square_root"\n'
                    )

        allowed_token = []
        for name in func_list:
            allowed_token.extend(model.encode(name))
        allowed_token.extend(model.encode(" end\n"))
        allowed_token = list(set(allowed_token))
        func_lookup = {f['name']: f for f in raw_funcs.functions}
        for p in prompts:
            func = ""
            functions = func_list
            encoded = func_prompt + model.encode(f'prompt: {p}\n' + '"answer": "')
            for _ in range(100):
                print(func)
                if func:
                    functions = [f for f in functions if func in f]
                if len(functions) == 1:
                    func = functions[0]
                    funcs.append(
                            Func(
                                name=functions[0],
                                description=func_lookup[func]['description'],
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
                func += text
                if '"' in func:
                    func = func.replace('"', "").strip()
                    print(func)
                    funcs.append(
                            Func(name=func,
                                 description=func_lookup[func]['description'],
                                 prompt=p
                                 )
                                 )
                    encoded.extend(model.encode("\n"))
                    break
                encoded.append(new_id)
        model.del_model()
        model.set_module("coder")

        param_prompt = model.encode(
                'function name: fn_add_numbers\n'
                "description: Add two numbers together and return their sum."
                'expected parameters: "a": "number", "b": "number"\n'
                "prompt: add 9 and 5\n"
                'parameters: "a": "9", "b": "5"\n\n'
                "function name: fn_substitute_string_with_regex\n"
                'description: Replace all occurrences matching a regex pattern in a string.\n'
                'expected parameters: "source_string": "string", "regex": "string", "replacement": "string"\n'
                'prompt: replace all a and b in this sentence "hello i am born to be here with you as a pawn" with z\n'
                'parameters: "source_string": "hello i am born to be here with you as a pawn", "regex": "[ab]+", "replacement": "z"\n\n'
                "function name: fn_substitute_string_with_regex\n"
                'description: Replace all occurrences matching a regex pattern in a string.\n'
                'expected parameters: "source_string": "string", "regex": "string", "replacement": "string"\n'
                'prompt: subtitute every number with a dash in this sentence "hello 2 i0am happ7y to be 328here"\n'
                'parameters: "source_string": "hello 2 i0am happ7y to be 328here", "regex": "\\d+", "replacement": "-"\n\n'
                )
        for f in funcs:
            expected_param = " ,".join([f'"{k}": "{v["type"]}"' for k, v in func_lookup[f.name]['parameters'].items()])
            prompt = (
                f"function name: {f.name}\n"
                f"description:{ f.description}\n"
                f"expected parameters: {expected_param}\n"
                f"prompt: {f.prompt}\n"
                'parameters: '
                )
            encoded = param_prompt + model.encode(prompt)
            for param in func_lookup[f.name]["parameters"]:
                encoded.extend(model.encode(f'"{param}": "'))
                param_val = ""
                for _ in range(200):
                    logits = model.model.get_logits_from_input_ids(encoded)
                    new_id = int(np.argmax(logits))
                    param_val += model.decode([new_id])
                    encoded.append(new_id)
                    print(param_val)
                    if '"' in param_val:
                        break

                clean_val = param_val[:param_val.find('"')]
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
