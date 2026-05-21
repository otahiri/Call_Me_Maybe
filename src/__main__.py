from llm_sdk import Small_LLM_Model
import json
import numpy as np
import torch


def test(prompt: str):
    model = Small_LLM_Model()
    encoded = model.encode(prompt)

    max_token = 200

    for _ in range(max_token):
        logits = model.get_logits_from_input_ids(encoded.squeeze(0).tolist())
        new_id = np.argmax(np.array(logits))
        token_text = model.decode(torch.tensor(new_id))
        print(token_text, end='')
        new_id_tensor = torch.tensor([[new_id]], device=encoded.device)
        encoded = torch.cat((encoded, new_id_tensor), dim=1)
    print()


def main():
    prompt = []
    with open("data/input/function_calling_tests.json", "r") as file:
        data = json.load(file)
    for key in data:
        for value in key.values():
            prompt.append(value)
    test(prompt[1])


if __name__ == "__main__":
    main()
