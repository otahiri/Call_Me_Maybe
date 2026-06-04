from llm_sdk import Small_LLM_Model
import json
import numpy as np
import torch


def test(prompt: str):
    model = Small_LLM_Model()
    prompt = (
            "You are a strict data extraction script. Your ONLY job is to identify the core action or function requested in the user's input and output the function name in snake_case followed immediately by a space and the word \"end\"."
            ""
            "RULES:"
            "1. Output exactly the function name and the word \"end\"." 
            "2. Do NOT output any other text, no explanations, no punctuation, no \"Sure\", and no greetings."
            "3. If the action is unclear, output: unknown_action end"
            ""
            "EXAMPLES:"
            "Input: \"can you turn off the living room lights for me?\""
            "Output: turn_off_lights end"
            ""
            "Input: \"what's the weather like in Tokyo right now?\""
            "Output: get_weather end"
            ""
            "Input: \"create a new file called test.txt\""
            "Output: create_file end"
            ""
            "Input: \"fetch me the user details for ID 45\""
            "Output: get_user_details end"
            ""
            f"Input: \"{prompt}\""
            "Output:"
)
    encoded = model.encode(prompt)

    max_token = 200
    answer = ""

    for _ in range(max_token):
        logits = model.get_logits_from_input_ids(encoded.squeeze(0).tolist())
        new_id = np.argmax(np.array(logits))
        token_text = model.decode(torch.tensor(new_id))
        answer += token_text
        print(token_text, end='')
        if "end" in answer:
            break
        new_id_tensor = torch.tensor([[new_id]], device=encoded.device)
        encoded = torch.cat((encoded, new_id_tensor), dim=1)
    print()


def main():
    prompts = []
    with open("data/input/function_calling_tests.json", "r") as file:
        data = json.load(file)
    for key in data:
        for value in key.values():
            prompts.append(value)
    for prompt in prompts:
        test(prompt)


if __name__ == "__main__":
    main()
