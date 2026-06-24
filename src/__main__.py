from pathlib import Path
import sys
import gc

try:
    from typing import Any
    import argparse

    import pydantic
    from src import RawInputs, ModelInterface, Func

    import json
    import numpy as np

    def run_model() -> None:
        prompts = []
        model = ModelInterface()
        model.set_module("base")
        data: Any
        func_def: Any
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
        try:
            with open(args.functions_definition, "r") as func_file:
                func_def = json.load(func_file)
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            raise ValueError("could not access the function definition file")
        except json.decoder.JSONDecodeError:
            raise ValueError("could not decode func def file")

        try:
            with open(args.input, "r") as file:
                data = json.load(file)
                for line in data:
                    for key, value in line.items():
                        if not value:
                            raise ValueError("empty prompt detected")
                        if key == "prompt":
                            prompts.append(value)
                        else:
                            raise ValueError(
                                    'error prompts should be \
{"prompt": prompt text}')
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            raise ValueError("could not acces prompt file")
        except json.decoder.JSONDecodeError:
            raise ValueError("could not decode prompt input file")

        raw_funcs = RawInputs(functions=func_def, prompts=prompts)
        funcs = []

        func_list = [f['name'] for f in raw_funcs.functions]
        func_prompt = model.encode(
                f"function list = [{' ,'.join(func_list)}]\n"
                "prompt: add 2 and 3\n"
                '"answer": "fn_add_numbers"\n'
                'prompt: replace all number in this sentence '
                '"hello my name is me and 31 i 213 am 65 89 years old" '
                'with "-"\n'
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
        for p in raw_funcs.prompts:
            func = ""
            functions = func_list
            encoded = func_prompt + model.encode(
                    f'prompt: {p}\n' + '"answer": "')
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
        model.set_module("coder")

        param_prompt = model.encode(
                'function name: fn_add_numbers\n'
                "description: Add two numbers together and return their sum."
                'expected parameters: "a": "number", "b": "number"\n'
                "prompt: add 9 and 5\n"
                'parameters: "a": "9.0", "b": "5.0"\n\n'
                'function name: fn_add_numbers\n'
                "description: Add two numbers together and return their sum."
                'expected parameters: "a": "integer", "b": "integer"\n'
                "prompt: add 8 and 2\n"
                'parameters: "a": "8", "b": "2"\n\n'
                "function name: fn_substitute_string_with_regex\n"
                'description: Replace all occurrences matching a regex '
                'pattern in a string.\n'
                'expected parameters: "source_string": "string", '
                '"regex": "string", "replacement": "string"\n'
                'prompt: subtitute every number with a dash in this'
                'sentence "hello 2 i0am happ7y to be 328here"\n'
                'parameters: "source_string": "hello 2 i0am happ7y'
                ' to be 328here", "regex": "\\d+", "replacement": "-"\n\n'
                )
        for f in funcs:
            expected_param = " ,".join(
                    [f'"{k}": "{v["type"]}"'
                     for k, v in func_lookup[f.name]['parameters'].items()]
                    )
            prompt = (
                f"function name: {f.name}\n"
                f"description:{ f.description}\n"
                f"expected parameters: {expected_param}\n"
                f"prompt: {f.prompt}\n"
                'parameters: '
                )
            encoded = param_prompt + model.encode(prompt)
            for name, param_type in func_lookup[f.name]["parameters"].items():
                encoded.extend(model.encode(f'"{name}": "'))
                param_val = ""
                for _ in range(200):
                    logits = model.model.get_logits_from_input_ids(encoded)
                    new_id = int(np.argmax(logits))
                    param_val += model.decode([new_id])
                    encoded.append(new_id)
                    print(param_val)
                    if '"' in param_val:
                        break

                clean_val: str | int | float = param_val[:param_val.find('"')]
                param_type = param_type["type"]
                print(param_type)
                if param_type == "integer":
                    clean_val = int(clean_val)
                elif param_type == "number":
                    clean_val = float(clean_val)
                else:
                    clean_val = clean_val
                f.parameters[name] = clean_val
                print(f)
        gc.collect()
        mode = "w" if Path(args.output).exists() else "x"
        with open(args.output, mode) as out_fp:
            json.dump([f.model_dump() for f in funcs], out_fp, indent=2)

    def cli() -> None:
        try:
            run_model()
        except pydantic.ValidationError as e:
            print(e.errors()[0]["msg"], file=sys.stderr)
        except Exception as e:
            print(e, file=sys.stderr)

    if __name__ == "__main__":
        cli()
except KeyboardInterrupt:
    print("user force stopped the program", file=sys.stderr)
