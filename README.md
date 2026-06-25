# Call_Me_Maybe
This project has been created as part of the 42 curriculum by otahiri-.


**DESCRIPTION:
    a project about function calling.
    usually when we ask an llm a question it will answers regardless of whether it know or not,
    this was a huge issue back in the days, chatgpt would get the most basic mathmatical questions wrong,
    to solve this issue a new technique was introduced function calling,
    it is a way for the llm to cheat and use external tools to execute actions using pre-defined functions, this include but not limited any and all mathmatical questions, web scraping, scan a photo for text etc...
    so the way it works is you give the ai a json file containing all the function definition, so when you ask it a question it determines the correct function.
    then it extract the parameters from the prompt, this makes it able to answer complicated questions


**INSTRUCTIONS:
    to test lint run the command:
            ```bash
                make lint
            ```

    to install the dependencies run the command:
    ```bash
        make install
    ```

    to run the program run the command:
    ```bash
        make run
    ```

    additionally you provide custome input files with certain flags:
        to use your own functions definition file use the flag --functions_definition "path to function definition file" (by default it is data/input/functions_definition.json)
        to use your own function test prompt file use the flag --input "path to function calling file" (by default is it data/input/function_calling_tests.json)
        to use your own out put file use the flag --output "path to save output file" (by default it is data/output/function_calls.json)
        Note: all flags are optional and if any flag is not provided the default value is used

    to run the program in debug mode run the command:
        ```bash
            make debug
        ```
    to run tests for the program run the command:
    ```bash
        make test
    ```

    to remove cache files run the command:
    ```bash
        make clean
    ```

**Resources:

    [let's make chatgpt from scract](https://youtu.be/kCc8FmEb1nY?si=cGOiCjKdMzG8Jtot): one of the best way to start your ai journey and understand the hidden details of how llms works
    [make gpt tokenizer](https://youtu.be/zduSFxRajkE?si=4bZ7imegGyblIfjl): the best guide and tutorial to make your own tokenizer


**Algorithm explanation:
    the way i approached constrained decoding in this project is by taking the logits given by the llm and using the function np.full_like, it can clone the list of logits, additionally you can give it the value to set in place of all values, i set it to negative infinity, then i set only the allowed tokens in the masked list to the value in the original logits list, and set the the logits to the masked logits, this makes the invalid values be irrelevant which makes the llm work more efficently,


**Design decisions:
    to make the usage of the model easier i made a class called ModelInterface, this has useful methods such as encode and decode which i impemented to make the concepts easier to understand,
    set_model which set the model to the model value the specified version, by default it is the Qwen/Qwen3-0.6B but if you set it to code you get the Qwen/Qwen2.5-Coder-0.5B. those are the only supported models currently


**Performance analysis: 
    since the llm takes longer when provided with a beefy prompt i chose to go with examples, so i give it a sample input and expected output, this give the llm simple to follow pattern, this made the llm way faster and more reliable,
    additionally to make it follow the exact output format, i force it to follow my pattern so like name: " and wait for it to generate the closing quotes, this makes early exit easier and guarantees the output to be better than normal.


**Challenges faced:
    the main issue was how unreliable the base mode at extracting the parameters, to solve this i had to use another model for parameters extraction, this was the biggest improvment to the output.


**Testing strategy:
    
