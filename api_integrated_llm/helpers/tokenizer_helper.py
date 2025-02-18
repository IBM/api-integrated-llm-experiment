from transformers import AutoTokenizer

tokenizer = ""


def get_granite_tokenizer():
    global tokenizer
    if tokenizer:
        return tokenizer
    else:
        BASE_MODEL = "ibm-granite/granite-3.1-8b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        return tokenizer


def granite_prompt_input(input, function, example_str, base_prompt: str):
    prompts_initial = {"role": "user", "content": input}
    extra_turn = {
        "role": "system",
        "content": (
            'DO NOT try to answer the user question, just invoke the tools needed to respond to the user, if any. The output MUST strictly adhere to the following JSON format: [{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}, "label": "$var_1"}, ... (more tool calls as required)]. Please make sure the parameter type is correct and follow the documentation for parameter format. If no function call is needed, please directly output an empty list.\nHere are some examples:\n'
            + example_str
            + "\n"
        ),
    }
    prompts = [extra_turn] + [prompts_initial]
    tokenizer = get_granite_tokenizer()
    formatted_prompt = tokenizer.apply_chat_template(
        prompts, function, tokenize=False, add_generation_prompt=True
    )

    formatted_prompt = base_prompt + formatted_prompt

    return formatted_prompt
