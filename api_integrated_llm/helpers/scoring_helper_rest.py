import re
import regex
import json
# from api_integrated_llm.helpers.output_parsers import get_output_list
from typing import Any, Dict, List

from api_integrated_llm.data_models.source_models import EvaluationOutputResponseDataUnit
from api_integrated_llm.helpers.file_helper import (
    get_json_dict_from_txt,
)


def get_output_list(
    prediction,
) -> List[Dict[str, Any]]:
    return get_json_dict_from_txt(txt=prediction.output) if isinstance(prediction.output, str) else list(map(lambda unit: unit.model_dump(), prediction.output))  # type: ignore


def parse_functions(string_input):
    # 1. Strip the leading/trailing square brackets
    stripped = string_input.strip("[]")

    # 2. Split by `),` but keep the closing parenthesis attached to each part
    chunks = [chunk.strip() for chunk in stripped.split("),")]

    results = []
    pattern = re.compile(r'^(?P<name>\w[\w_]*)\((?P<args>.*)\)$')

    for chunk in chunks:
        chunk = chunk.strip()
        # Add back the ) if we lost it during splitting
        if not chunk.endswith(')'):
            chunk += ')'

        match = pattern.match(chunk)
        if not match:
            continue

        func_name = match.group('name')
        args_str = match.group('args').strip()

        args_dict = {}
        if args_str:  # parse the arguments k1=v1, k2=v2 ...
            arg_pairs = [arg.strip() for arg in args_str.split(',')]
            for pair in arg_pairs:
                if '=' in pair:
                    key, val = pair.split('=', 1)
                    key = key.strip()
                    val = val.strip()
                    # Convert numbers if needed
                    if val.isdigit():
                        val = int(val)
                    args_dict[key] = val

        results.append({
            'name': func_name,
            'arguments': args_dict
        })
    return results

def parse_llama_3_70b_instruct_rest(
    prediction,
    num_errors_parsing_pred_intent: int,
    is_single_intent_detection: bool = True,
    skip_grounding: bool = True,
):
    item = prediction.model_dump()
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []  # type: ignore
    gold_dict_list = get_output_list(prediction=prediction)
    parsing_error_messages: List[str] = []
    new_item = {"name": item["output"][0]["name"], "arguments": item["output"][0]["arguments"]}
    gold_dict_list = [new_item]
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]
   
    pred = item["generated_text"].strip()
    try:
        pred_dict_list = parse_functions(pred)
        pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        pred_dict_list_new = []
        for pred_fun_call in pred_dict_list:
            args = {}

            if "arguments" in pred_fun_call:
                args = pred_fun_call["arguments"]
                new_arg = {}
                for key_arg, arg_val in args.items():
                    try:
                        new_arg[key_arg] = json.loads(arg_val)
                    except:
                        if isinstance(arg_val, str):
                            arg_val = arg_val.replace("'", "")
                        new_arg[key_arg] = arg_val
                args = new_arg
            pred_dict_list_new.append({"name": pred_fun_call["name"], "arguments": args}) 
        pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        pred_func_calls = [json.dumps(func) for func in pred_dict_list_new]
    except Exception as e:
        parsing_error_messages = []
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
        parsing_error_messages,
    )

def extract_inner_content(input_str, regex_expr=r'\[?<tool_call>\[(.*)\]?\s*$'):
    """
    Extracts the content inside the outermost double quotes.
    For an input like:
    [<tool_call>["<content>"]]
    """
    m = regex.search(regex_expr, input_str)
    if m:
        return m.group(1)
    return ""

def ground_seq_nested_repsonse(api_list):
    def check_label_in_slot(label, slot_v):
        if slot_v.startswith("$var"):
            if "." in slot_v:
                lbl_slot = slot_v.split(".", 1)[0].replace("$", "")
                if lbl_slot == label:
                    # if lbl_slot == 'var_10': ipdb.set_trace()
                    return True
        return False

    # ipdb.set_trace()
    # label_api_map = {api['name']: api['label'] for api in api_list if not api['name'] == 'var_result'}
    label_api_map = {}
    for api in api_list:
        if api["name"] == "varResult":
            continue
        if api["name"] == "var_result":
            continue
        if "label" in api:
            if api["label"] is None:
                api["label"] = ""
            # label_api_map[api['name']] = api['label']
            lbl = api["label"].replace("$", "")
            # if lbl == 'var_10': ipdb.set_trace()
            label_api_map[lbl] = api["name"]

    grounded_api_list = []
    # ipdb.set_trace()
    for api in api_list:
        if api["name"] == "var_result":
            continue
        temp_arguments = {}
        if "arguments" in api:
            arg_dict = api["arguments"]
        elif "parameters" in api:
            arg_dict = api["parameters"]
        else:
            arg_dict = {}
        for s_n, s_v in arg_dict.items():
            for l, a in label_api_map.items():  # noqa: E741
                # if type(s_v) == str and l in s_v:
                if type(s_v) == str and check_label_in_slot(l, s_v):
                    # ipdb.set_trace()
                    s_v = s_v.replace(l, a)
                elif type(s_v) == list:
                    new_s_v = []
                    for v in s_v:
                        # if type(v) == str and l in v:
                        if type(v) == str and check_label_in_slot(l, v):
                            # ipdb.set_trace()
                            v = v.replace(l, a)
                        # elif type(v) == dict and l in json.dumps(v):
                        elif type(v) == dict and check_label_in_slot(l, json.dumps(v)):
                            # ipdb.set_trace()
                            v = json.loads(json.dumps(v).replace(l, a))
                        new_s_v.append(v)
                    s_v = new_s_v
                    # break
            temp_arguments[s_n] = s_v

        grounded_api_list.append({"name": api["name"], "arguments": temp_arguments})

    return grounded_api_list



def parse_mixtral_output_rest(prediction, num_errors_parsing_pred_intent, skip_grounding=True):
    item = prediction.model_dump()
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []

    new_item = {"name": item["output"][0]["name"], "arguments": item["output"][0]["arguments"]}
    gold_dict_list = [new_item]
    # gold_dict_list = item["output"]
    
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    ## Pred
    try:
        gen_text = item["generated_text"].strip()
        if gen_text.endswith("'"):
            gen_text = gen_text[:-1]
        if not gen_text.startswith("["):
            gen_text = "[" + gen_text
        if not gen_text.endswith("]"):
            gen_text = gen_text + "]"

            # default
        try:
            pred_dict_list = json.loads(gen_text)
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        except Exception as e:
            print(e)
            # Step 1: Extract the inner content and unescape it.
            inner_content = extract_inner_content(gen_text, r'\[?<tool_call>\[(.*)\]?\s*$')

            unescaped = inner_content.encode().decode('unicode_escape')
            # Step 2: Use finditer to capture all JSON objects.
            pred_dict_list = []
            for m in pattern.finditer(unescaped):
                name = m.group("name")
                args_str = m.group("args")
                # Parse the "arguments" string to a Python object.
                try:
                    arguments = json.loads(args_str)
                    # If the arguments object is empty, return an empty string.
                    if arguments == {}:
                        arguments = {}
                except Exception as e:
                    # Fallback: if parsing fails, keep the raw string.
                    arguments = args_str
                    num_errors_parsing_pred_intent += 1
                pred_dict_list.append({"name": name, "arguments": arguments})

            if skip_grounding:
                pred_func_calls = [json.dumps(func) for func in pred_dict_list]


    except Exception as e:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors, []
    )


def parse_deepseek_output_rest(prediction, num_errors_parsing_pred_intent, skip_grounding=True, model_name=None):
    
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    # Convert a single instance to dict
    item = prediction.model_dump()
    
    new_item = {"name": item["output"][0]["name"], "arguments": item["output"][0]["arguments"]}
    gold_dict_list = [new_item]
    # gold_dict_list = item["output"]
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    ## Pred
    try:
        gen_text = item["generated_text"].strip()

        if gen_text.endswith("'"):
            gen_text = gen_text[:-1]
        if model_name not in "qwen":
            if not gen_text.startswith("["):
                gen_text = "[" + gen_text
            if not gen_text.endswith("]"):
                gen_text = gen_text + "]"

        if "deepseek" in model_name or "gpt" in model_name:

            pattern = r"```json\s*(.*?)\s*```"
        elif "hammer" in model_name:
            pattern = r"```\s*(.*?)\s*```"
        elif "qwen" in model_name:
            pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        match = re.search(pattern, gen_text, flags=re.DOTALL)
        if match:
            extracted_text = match.group(1)
        else:
            print("No match found.") 

        if "qwen" not in model_name:
            pred_dict_list = json.loads(extracted_text)
        else:
            pred_dict_list = [extracted_text]

        if "qwen" not in model_name and len(pred_dict_list)!=0 and not isinstance(pred_dict_list[0], dict):
            pred_dict_list = []

        if "qwen" not in model_name:   
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = [func for func in pred_dict_list]
    except Exception as e:

        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors, []
    )

def parse_llm_out_rest_dataset(prediction, num_errors_parsing_pred_intent, is_single_intent_detection=True,skip_grounding=True):
    item = prediction.model_dump()
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    new_item = {"name": item["output"][0]["name"], "arguments": item["output"][0]["arguments"]}
    gold_dict_list = [new_item]

    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    pattern = regex.compile(
        r'\{"name":\s*"(?P<name>[^"]+)"\s*,\s*"arguments":\s*(?P<args>\{(?:[^{}]+|(?R))*\})\}'
    )

    try:
        gen_text = item["generated_text"].strip()
        if gen_text.endswith("'"):
            gen_text = gen_text[:-1]
        if not gen_text.startswith("["):
            gen_text = "[" + gen_text
        if not gen_text.endswith("]"):
            gen_text = gen_text + "]"
        # Step 1: Extract the inner content and unescape it.
        inner_content = extract_inner_content(gen_text)

        unescaped = inner_content.encode().decode('unicode_escape')

        # Step 2: Use finditer to capture all JSON objects.
        pred_dict_list = []
        for m in pattern.finditer(unescaped):
            name = m.group("name")
            args_str = m.group("args")
            # Parse the "arguments" string to a Python object.
            try:
                arguments = json.loads(args_str)
                # If the arguments object is empty, return an empty string.
                if arguments == {}:
                    arguments = {}
            except Exception as e:
                # Fallback: if parsing fails, keep the raw string.
                arguments = args_str
                num_errors_parsing_pred_intent += 1
            pred_dict_list.append({"name": name, "arguments": arguments})

        # Output the results.
        # print(json.dumps(pred_dict_list, indent=2))
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = (
                ground_seq_nested_repsonse(pred_dict_list)
                if "label" in item["generated_text"]
                else pred_dict_list
            )
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    parsing_error_messages = []
    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
        parsing_error_messages
    )


def parse_agent_rest(
    prediction,
    num_errors_parsing_pred_intent: int,
    is_single_intent_detection: bool = True,
    skip_grounding: bool = True,
):
    item = prediction.model_dump()
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []  # type: ignore
    gold_dict_list = get_output_list(prediction)
    parsing_error_messages: List[str] = []
    new_item = {"name": item["output"][0]["name"], "arguments": item["output"][0]["arguments"]}
    gold_dict_list = [new_item]
    # if skip_grounding:
    gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    # else:
    #     gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
    #     gold_func_calls = [json.dumps(func) for func in gold_func_calls]
    pred = item["generated_text"].strip()
    try:
        pred_dict_list = json.loads(pred)
        pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        # pred_func_calls = pred_dict_list
    except Exception as e:
        parsing_error_messages = []
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True
    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
        parsing_error_messages,
    )

