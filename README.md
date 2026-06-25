# Call_Me_Maybe
*This project has been created as part of the 42 curriculum by otahiri-.*


## Description

Call_Me_Maybe is a function-calling project that turns natural language prompts into structured JSON function calls.
Instead of answering the prompt directly, the program selects the best function from a provided catalog and extracts the
arguments needed to call it.

The goal is to keep the output valid, structured, and predictable by using constrained decoding rather than free-form text generation.

## Instructions

### Requirements

- Python 3.10+
- `uv`

### Install

```bash
make install
```

### Run

```bash
make run
```

By default, the program reads:

- `data/input/functions_definition.json`
- `data/input/function_calling_tests.json`

and writes:

- `data/output/function_calls.json`

### Custom paths

```bash
uv run python -m src \
  --functions_definition [path to functions_definition.json] \
  --input [path to input.json] \
  --output [path to output.json]
```
 or using the make commands
 ```bash
  make ARGS=" --functions_definition [path to functions_definition.json]\
  --input [path to input.json]\
  --output [path to output.json]"
 ```

### Other useful commands

```bash
make lint
make test
make debug
```

## Algorithm explanation

The solution uses a two-stage constrained decoding pipeline:

1. **Function selection**: constrained decoding is used to mask valid tokens and set the rest to -inf,
    each iteration the output is compared to the every element in list of function names, and the names that don't match the output or removed.
    once there is a single element in the list that name is selected as the correct answer

2. **Parameter extraction**: once the function is chosen, the program switches to the coder model and generates each
   parameter value one field at a time. Numeric fields are restricted to digit-like tokens plust double quotes, while string fields are left
   more open but still decoded in a controlled way.
   the program keeps getting the next toeken until a double quote is generated to segnify the end of the parameter.

This approach narrows the search space so the model stays close to the expected JSON structure instead of producing
unconstrained text.

## Design decisions

- **Two model phases**: one phase for function routing and one phase for parameter extraction.
- **Few-shot prompting**: example prompt/answer to help the model get into the function/parameters extracting mode.
- **Token masking**: only valid tokens are considered during sensitive parts of decoding.
- **Strict input validation**: function definitions and prompts are validated before generation starts.
- **JSON output only**: results are serialized as JSON objects so downstream tools can consume them directly.

## Performance analysis

- **Accuracy**: constraining the token space improves consistency and reduces invalid function names or malformed values.
- **Speed**: greedy decoding is simple and fast, but it still requires multiple forward passes per generated token.
- **Reliability**: the more the output is constrained, the less likely the model is to drift away from the expected format.

The main trade-off is that stronger constraints improve structure but can limit flexibility for ambiguous prompts.

## Challenges faced

- Formatting the prompt to guide the llm into the correct path.
- Keeping function names valid while still allowing the model to pick the right one, without taking too long.
- Handling different parameter types such as string, integer, and number.

These issues were handled by combining prompt examples, token filtering, and step-by-step extraction.

## Testing strategy

Validation was done with a mix of automated and manual checks:

- input validation tests for missing files, empty prompts, and malformed JSON
- fixture-based checks for tokenizer and merge loading
- command-line execution tests through the provided test suite
- output inspection to confirm the generated file is valid JSON and matches the expected structure

## Example usage

```bash
uv run python -m src
```

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

Example output shape:

```json
[
  {
    "name": "[function name]",
    "description": "[function description]",
    "prompt": "[original prompt]",
    "parameters": {
      "[parameter name]": "[parameter value]"
    }
  }
]
```

## Resources

- intro to an llm: [video](https://youtu.be/zjkBMFhNj_g?si=5V11WJb4f-NG20Wq)
- make chatgpt from scratch: [video](https://youtu.be/kCc8FmEb1nY?si=9txE2MBTplbo2Vo-)
- make gpt tokenizer: [video](https://youtu.be/zduSFxRajkE?si=XM0qIT7RGAf72iy2)

### AI usage

AI was used to help draft and organize this README, summarize the implementation into the required sections, and polish the
wording for clarity.
additionally AI was used for research and minimal debugging

