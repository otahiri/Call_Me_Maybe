"""
    function calling program
    relies on constrained decoding to generate the name of
    parameters to solve the issue a prompt propose
"""
try:
    from pathlib import Path
    import sys
    import gc
    from typing import Any
    import argparse

    import pydantic
    from src import RawInputs, ModelInterface, Func, set_module

    import json
    import numpy as np

    def get_func_name(func_list: list,
                      model: ModelInterface, func_prompt: list,
                      prompt: str, func_lookup: dict,
                      allowed_token: list) -> Func:
        """process individual prompts and extract func name from it

        Args:
            func_list: list of name of all functions
            model: model interface
            func_prompt: default additional starting part of all prompts
            prompt: prompt to be processed
            func_lookup: dictionary to look up functions using a name
            allowed_token: allowed token of functions names

        Returns:
            [TODO:return]
        """
        func: Func = Func(name="dummy", prompt="dummy", parameters=dict())
        name = ""
        functions = func_list
        encoded = func_prompt + model.encode(
                f'prompt: {prompt}\n' + '"answer": "')
        for _ in range(100):
            if name:
                functions = [f for f in functions if name in f]
            if len(functions) == 1:
                name = functions[0]
                func = Func(name=functions[0],
                            description=func_lookup[name]['description'],
                            prompt=prompt)
                break
            logits = model.model.get_logits_from_input_ids(encoded)
            mask = np.full_like(logits, -np.inf)
            for token in allowed_token:
                mask[token] = logits[token]
            logits = mask.tolist()
            new_id = int(np.argmax(logits))
            text = model.decode([int(new_id)])
            name += text
            if '"' in name:
                name = name.replace('"', "").strip()
                func = Func(name=functions[0],
                            description=func_lookup[name]['description'],
                            prompt=prompt)
                break
            encoded.append(new_id)
        return func


    def process_prompts(model: ModelInterface, func_lookup: dict,
                        raw_funcs: RawInputs) -> list[Func]:
        """process prompts and extract function name

        Args:
            model: model interface
            func_lookup: dictionary to look up function from the name
            prompts: prompt to be processed

        Returns:
            list if Func object containing name and prompt
        """
        funcs: list[Func] = []

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
        for p in raw_funcs.prompts:
            funcs.append(get_func_name(func_list, model, func_prompt, p,
                                       func_lookup, allowed_token))
        return funcs

    def extract_params(model: ModelInterface,
                       func_lookup: dict, func: Func,
                       param_prompt: list, numeral_tokens: list):
        """extract parameters from a single prompt

        Args:
            model: model interface
            func_lookup: dictionary to look up functions from name
            func: function object containing info the function
            param_prompt: default encoded prompt for all params
            numeral_tokens: numeral tokens to use incase the parameter
            is a number or integer
        """
        expected_param = " ,".join(
                [f'"{k}": "{v["type"]}"'
                 for k, v in func_lookup[func.name]['parameters'].items()]
                )
        prompt = (
                f"function name: {func.name}\n"
                f"description:{ func.description}\n"
                f"expected parameters: {expected_param}\n"
                f"prompt: {func.prompt}\n"
                'parameters: '
                )
        encoded = param_prompt + model.encode(prompt)
        for name, param_type in func_lookup[func.name]["parameters"].items():
            encoded.extend(model.encode(f'"{name}": "'))
            param_val = ""
            for _ in range(200):
                logits = model.model.get_logits_from_input_ids(encoded)
                if param_type in ["number", "integer"]:
                    mask = np.full_like(logits, -np.inf)
                    for token in numeral_tokens:
                        mask[token] = logits[token]
                    logits = mask
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
            func.parameters[name] = clean_val
            print(func)

    def process_parameters(model: ModelInterface, funcs: list[Func],
                           func_lookup: dict) -> None:
        """loop through all prompts and extract parameters

        Args:
            model: model interface
            funcs: function list
            func_lookup: dictionary to get function from name
        """
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
        numeral_tokens = model.encode('.0123456789"')
        for func in funcs:
            extract_params(model, func_lookup,
                           func, param_prompt, numeral_tokens)

    def run_model() -> None:
        """run the program from using the input 
        prompts and function definitions

        Raises:
            ValueError: when an invalid input is detected
        """
        prompts = []
        model = ModelInterface()
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
        func_lookup = {f['name']: f for f in raw_funcs.functions}

        funcs = process_prompts(model, func_lookup, raw_funcs)

        model.model = set_module("coder")
        process_parameters(model, funcs, func_lookup)

        gc.collect()
        mode = "w" if Path(args.output).exists() else "x"
        try:
            with open(args.output, mode) as out_fp:
                json.dump([f.model_dump() for f in funcs], out_fp, indent=2)
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            raise ValueError("cannot open output json file")

    def cli() -> None:
        """main function to run project"""
        try:
            run_model()
        except pydantic.ValidationError as e:
            print(e.errors()[0]["msg"], file=sys.stderr)

    if __name__ == "__main__":
        cli()

except KeyboardInterrupt:
        print("user force stopped the program", file=sys.stderr)
