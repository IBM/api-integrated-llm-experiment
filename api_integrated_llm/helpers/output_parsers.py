import json
import re
from typing import Any, Dict


def get_deli_sep_str_list(text, deli=","):
    def find(s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    comma_indexes = find(text, deli)
    valid_comma_indexes = []

    for idx in comma_indexes:
        valid_flag = True
        lfs, _ = text[:idx], text[idx + 1 :]  # noqa: E203

        # Delimeter not inside quotes
        quotes_count_lfs = lfs.count('"')
        if not quotes_count_lfs % 2 == 0:
            valid_flag = False

        if valid_flag:
            valid_comma_indexes.append(idx)
    parts = []
    temp_idx = 0
    for idx in valid_comma_indexes:
        parts.append(text[temp_idx:idx])
        temp_idx = idx + 2
    parts.append(text[temp_idx:])
    parts = [p.strip() for p in parts]
    return parts


def process_slot_value(slot_value):
    if slot_value.startswith("'"):
        slot_value = slot_value[1:]
    if slot_value.endswith("'"):
        slot_value = slot_value[:-1]
    if not slot_value.startswith('"'):
        slot_value = '"' + slot_value
    if not slot_value.endswith('"'):
        slot_value = slot_value + '"'
    sub_slot_value = slot_value[1:-1].replace('"', "'")
    return sub_slot_value


def ground_seq_nested_repsonse(api_list):
    def check_label_in_slot(label, slot_v):
        if slot_v.startswith("$var"):
            if "." in slot_v:
                lbl_slot = slot_v.split(".", 1)[0].replace("$", "")
                if lbl_slot == label:
                    # if lbl_slot == 'var_10': ipdb.set_trace()
                    return True
        return False

    # label_api_map = {api['name']: api['label'] for api in api_list if not api['name'] == 'var_result'}
    label_api_map = {}
    for api in api_list:
        if api["name"] == "varResult":
            continue
        if api["name"] == "var_result":
            continue
        if "label" in api:
            # label_api_map[api['name']] = api['label']
            lbl = api["label"].replace("$", "")
            # if lbl == 'var_10': ipdb.set_trace()
            label_api_map[lbl] = api["name"]

    grounded_api_list = []

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
                    s_v = s_v.replace(l, a)
                elif type(s_v) == list:
                    new_s_v = []
                    for v in s_v:
                        # if type(v) == str and l in v:
                        if type(v) == str and check_label_in_slot(l, v):
                            v = v.replace(l, a)
                        # elif type(v) == dict and l in json.dumps(v):
                        elif type(v) == dict and check_label_in_slot(l, json.dumps(v)):
                            v = json.loads(json.dumps(v).replace(l, a))
                        new_s_v.append(v)
                    s_v = new_s_v
                    # break
            temp_arguments[s_n] = s_v

        grounded_api_list.append({"name": api["name"], "arguments": temp_arguments})

    return grounded_api_list


def parse_granite_20b_function_calling_output(
    item, num_errors_parsing_pred_intent, skip_grounding=False
):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item["output"])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred = item["generated_text"].strip().replace("ASSISTANT", "").strip()
        pred_str_list = pred.split("<function_call>")
        pred_dict_list = [json.loads(p) for p in pred_str_list if p]
        pred_dict_list = [p for p in pred_dict_list if not p["name"] == "var_result"]
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = (
                ground_seq_nested_repsonse(pred_dict_list)
                if "label" in pred
                else pred_dict_list
            )
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
    )


def parse_granite_3_output(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item["output"])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        if "generated_text" in item:
            generated_text = item["generated_text"]
            start_idx = generated_text.find("[")
            end_idx = generated_text.rfind("]")
            if start_idx == -1 or end_idx == -1 or start_idx > end_idx:
                raise Exception("Parsing error")

            pred_dict_list = json.loads(
                generated_text[start_idx : end_idx + 1]  # noqa: E203
            )  # noqa: E203
        else:
            raise Exception("no generated_text field in item")

        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls_dict = (
                ground_seq_nested_repsonse(pred_dict_list)
                if "label" in item["generated_text"]
                else pred_dict_list
            )

            pred_func_calls = []
            for func in pred_func_calls_dict:
                pred_func_calls.append(json.dumps(func))
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
    )


def parse_llama_3_output(
    prediction: Dict[str, Any],
    num_errors_parsing_pred_intent: int,
    skip_grounding=False,
):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(prediction["output"])

    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        # see what is here
        gen_text = prediction["generated_text"].strip()
        if gen_text.endswith("'"):
            gen_text = gen_text[:-1]
        if not gen_text.startswith("["):
            gen_text = "[" + gen_text
        if not gen_text.endswith("]"):
            gen_text = gen_text + "]"

        pred_dict_list = json.loads(gen_text)
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = (
                ground_seq_nested_repsonse(pred_dict_list)
                if "label" in prediction["generated_text"]
                else pred_dict_list
            )
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
    )


## TODO: Merge llama parser
def parse_llama_3_70b_instruct(
    item, num_errors_parsing_pred_intent, skip_grounding=False
):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item["output"])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred = item["generated_text"].strip()
        pred_dict_list = json.loads(pred)
        # pred_func_calls = ground_seq_nested_repsonse(pred_dict_list)
        # pred_func_calls = [json.dumps(func) for func in pred_func_calls]
        pred_dict_list = [p for p in pred_dict_list if not p["name"] == "var_result"]
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = (
                ground_seq_nested_repsonse(pred_dict_list)
                if "label" in pred
                else pred_dict_list
            )
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        try:
            pred = item["generated_text"].strip()
            if pred.startswith("[") and pred.endswith("]"):
                pred = pred[1:-1]
                pred_list = pred.split("),")
                new_pred_list = []
                for p in pred_list:
                    if p.strip().endswith(")"):
                        new_pred_list.append(p)
                    else:
                        new_pred_list.append(p + ")")
                pred_func_calls = []
                for p in new_pred_list:
                    intent = p.split("(", 1)[0]
                    slot_str = p.split("(", 1)[1][:-1]
                    slots = get_deli_sep_str_list(slot_str)
                    arg_dict = {}

                    for s in slots:
                        s_n, s_v = s.split("=")[0].strip(), s.split("=")[1].strip()
                        arg_dict[s_n] = process_slot_value(s_v)

                    pred_func_calls.append(
                        json.dumps({"name": intent, "arguments": arg_dict})
                    )
            else:
                num_errors_parsing_pred_intent += 1
                pred_has_parsing_errors = True
        except:
            num_errors_parsing_pred_intent += 1
            pred_has_parsing_errors = True

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
    )


def parse_mistral_7b_instruct_v0_3(
    item, num_errors_parsing_pred_intent, skip_grounding=False
):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item["output"])

    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred = item["generated_text"].strip()
        pred_dict_list = json.loads(pred)

        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list)
            # pred_func_calls = pred_dict_list
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        try:
            pred = item["generated_text"].strip()
            pred_dict_list = json.loads(
                pred.replace("\n", "").replace("\_", "_")  # noqa: W605
            )  # noqa: W605
            if skip_grounding:
                pred_func_calls = [json.dumps(func) for func in pred_dict_list]
            else:
                pred_func_calls = ground_seq_nested_repsonse(pred_dict_list)
                # pred_func_calls = pred_dict_list
                pred_func_calls = [json.dumps(func) for func in pred_func_calls]
        except:
            num_errors_parsing_pred_intent += 1
            pred_has_parsing_errors = True

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
    )


def parse_hermes_2_pro_mistral_7B(
    item, num_errors_parsing_pred_intent, skip_grounding=False
):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item["output"])

    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred = item["generated_text"].strip()
        if pred.startswith("tool_call\n{"):
            pred = pred.replace("tool_call\n{", "<tool_call>\n{")

        func_str_list = re.findall(r"<tool_call>(.*?)</tool_call>", pred, re.DOTALL)

        pred_dict_list = []
        for p in func_str_list:
            try:
                pred_dict_list.append(json.loads(p.strip()))
            except:
                continue
        # pred_dict_list = [json.loads(p) for p in func_str_list]
        # print(pred_dict_list)

        assert len(pred_dict_list) > 0, "parsing issue"
        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list)
            # pred_func_calls = pred_dict_list
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
    )


def parse_xLAM_1b_fc_r(item, num_errors_parsing_pred_intent, skip_grounding=False):
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    ## Gold
    gold_dict_list = json.loads(item["output"])
    # gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
    # gold_func_calls = [json.dumps(func) for func in gold_func_calls]
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred = item["generated_text"].strip()
        pred = (
            pred.replace("[BEGIN OF ANSWER]", "")
            .replace("[BEGIN OF FUNCTION CALLS]", "")
            .replace("[BEGIN OF FUNCTION CALL]", "")
            .replace("[BEGIN OF ACTIONS]", "")
            .replace("[BEGIN OF ACTION]", "")
            .strip()
        )
        tool_dict = json.loads(pred)
        assert "tool_calls" in tool_dict, "parsing error"
        pred_dict_list = tool_dict["tool_calls"]

        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list)
            # pred_func_calls = pred_dict_list
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]
    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
    )


def parse_Hammer2_0_7b(
    item, num_errors_parsing_pred_intent, skip_grounding=False
):  # TODO: @Mayank
    """
    ### Input: 'item' as json object
    # Pred: Example from Granite3.0
    item['generated_text'] : '[{"name": "divide", "label": "$var_1", "arguments": {"arg_0": 45, "arg_1": 18}}, {"name": "divide", "label": "$var_2", "arguments": {"arg_0": 1, "arg_1": "$var_1.result$"}}]'
    # Gold:
    item['output'] : '[{"name": "multiply", "label": "$var_1", "arguments": {"arg_0": 18, "arg_1": 45}}, {"name": "divide", "label": "$var_2", "arguments": {"arg_0": "$var_1.result$", "arg_1": 27}}]'

    ### Output: List[String]
    pred_func_calls: ['{"name": "divide", "arguments": {"arg_0": 45, "arg_1": 18}}', '{"name": "divide", "arguments": {"arg_0": 1, "arg_1": "$divide.result$"}}']
    gold_func_calls: ['{"name": "multiply", "arguments": {"arg_0": 18, "arg_1": 45}}', '{"name": "divide", "arguments": {"arg_0": "$multiply.result$", "arg_1": 27}}']
    """

    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []

    ## Gold
    gold_dict_list = json.loads(item["output"])
    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls]

    ## Pred
    try:
        pred = item["generated_text"].replace("```", "").strip()
        pred_dict_list = json.loads(pred)
        assert len(pred_dict_list) > 0, "parsing issue"

        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls = ground_seq_nested_repsonse(pred_dict_list)
            pred_func_calls = [json.dumps(func) for func in pred_func_calls]

    except:
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
    )
