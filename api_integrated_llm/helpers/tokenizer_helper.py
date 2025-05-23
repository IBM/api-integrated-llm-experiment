from typing import List
from transformers import AutoTokenizer

from api_integrated_llm.data_models.source_models import ToolItemModel

tokenizer = ""


def get_granite_tokenizer():
    global tokenizer
    if tokenizer:
        return tokenizer
    else:
        BASE_MODEL = "ibm-granite/granite-3.1-8b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        return tokenizer


def granite_prompt_input(
    input: str,
    function: List[ToolItemModel],
    example_str: str,
    base_prompt: str,
    is_single_intent: bool = False,
) -> str:
    prompts_initial = {"role": "user", "content": input}

    if is_single_intent:
        extra_turn = {
            "role": "system",
            "content": (
                "Knowledge Cutoff Date: April 2024. Today's Date: February 19, 2025. You are Granite, developed by IBM. You are a helpful AI assistant with access to the following tools. When a tool is required to answer the user's query, respond with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist in the provided list of tools, notify the user that you do not have the ability to fulfill the request.<|end_of_text|>"
            ),
        }
    else:
        extra_turn = {
            "role": "system",
            "content": (
                'DO NOT try to answer the user question, just invoke the tools needed to respond to the user, if any. The output MUST strictly adhere to the following JSON format: [{"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}, "label": "$var_1"}, ... (more tool calls as required)]. Please make sure the parameter type is correct and follow the documentation for parameter format. If no function call is needed, please directly output an empty list.\n\n Relevant data is available in the file starting_table_var. \nHere are some examples:\n'
                + example_str
                + "\n"
            ),
        }
    prompts = [extra_turn] + [prompts_initial]
    tokenizer = get_granite_tokenizer()
    formatted_prompt = tokenizer.apply_chat_template(
        prompts,
        list(map(lambda item: item.model_dump(), function)),
        tokenize=False,
        add_generation_prompt=True,
    )

    formatted_prompt = base_prompt + formatted_prompt

    return formatted_prompt
