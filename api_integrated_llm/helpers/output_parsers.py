from copy import deepcopy
import json
from typing import Any, Dict, List, Optional, Tuple
import ast
from api_integrated_llm.data_models.common_models import CommonErrorModel
from api_integrated_llm.data_models.source_models import (
    EvaluationOutputResponseDataUnit,
)
from api_integrated_llm.helpers.file_helper import (
    get_json_data_with_two_step_parsing,
    get_json_dict_from_txt,
)


from api_integrated_llm.helpers.scoring_helper_rest import (
    parse_agent_rest,
    parse_llm_out_rest_dataset,
    parse_llama_3_70b_instruct_rest,
    parse_mixtral_output_rest,
    parse_deepseek_output_rest,
)


def get_deli_sep_str_list(text: str, deli: str = ",") -> List[str]:
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


def ground_seq_nested_repsonse(api_list) -> List[Dict[str, Any]]:
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
        if "label" in api and api["label"] is not None:
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
                            v = json.loads(
                                json.dumps(v).replace(l, a),
                            )
                        new_s_v.append(v)
                    s_v = new_s_v
                    # break
            temp_arguments[s_n] = s_v

        grounded_api_list.append({"name": api["name"], "arguments": temp_arguments})

    return grounded_api_list


def get_output_list(
    prediction: EvaluationOutputResponseDataUnit,
) -> List[Dict[str, Any]]:
    return get_json_dict_from_txt(txt=prediction.output) if isinstance(prediction.output, str) else list(map(lambda unit: unit.model_dump(), prediction.output))  # type: ignore


def resolve_ast_by_type(value) -> Any:
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]  # type: ignore
    elif isinstance(value, ast.Dict):
        output = {  # type: ignore
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)  # type: ignore
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)  # type: ignore
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))  # type: ignore
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output


def resolve_ast_call(elem) -> Dict[str, Any]:
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {"name": func_name, "arguments": args_dict}


def ast_parse(txt: str) -> List[Dict[str, Any]]:
    new_text = txt.strip("[]'")
    start_idx_list = new_text.find("[")
    end_idx_list = new_text.rfind("]")

    if (
        (len(new_text) > 0)
        and (start_idx_list != -1)
        and (end_idx_list != -1)
        and (end_idx_list > start_idx_list)
    ):
        new_text = new_text[start_idx_list : (end_idx_list + 1)]  # noqa: E203
        parsed = ast.parse(new_text, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return extracted
    raise Exception("No valid function calls for AST parser")


def manual_ast_parsing(txt: str) -> List[Dict[str, Any]]:
    if txt.startswith("[") and txt.endswith("]"):
        pred = txt[1:-1]
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

            pred_func_calls.append({"name": intent, "arguments": arg_dict})
        return pred_func_calls
    raise Exception("Maual AST parsing failed")


def manual_xml_parsing(txt: str) -> List[Dict[str, Any]]:
    if not ("<tool_call>" in txt and "</tool_call>" in txt):
        raise Exception("Parsing Error at manual_ast_parsing. No <tool_call>")
    function_calls: List[Dict[str, Any]] = []
    new_str = txt.replace("</tool_call>", "<tool_call>")
    new_str_list = new_str.split("<tool_call>")
    for segment in new_str_list:
        try:
            obj = get_json_data_with_two_step_parsing(  # type: ignore
                txt=segment, should_return_list=False
            )
            if obj is not None and isinstance(obj, dict):
                function_calls.append(deepcopy(obj))
        except Exception as e:
            error_message = str(e)
            print(error_message)
    return function_calls


def parse_multi_step(txt: str) -> List[Dict[str, Any]]:
    txt = txt.replace("\n", "").replace("\_", "_").strip()  # noqa: W605
    pred_dict_list: Optional[List] = None
    try:
        pred_dict_list = get_json_data_with_two_step_parsing(  # type: ignore
            txt=txt, should_return_list=True
        )
    except Exception as e:
        error_message_json = str(e)
        print(f"JSON parsing failed: {error_message_json}")

    if pred_dict_list is None:  # json parsing failed
        try:
            pred_dict_list = ast_parse(txt=txt)
        except Exception as e:
            error_message_ast = str(e)
            print(f"AST parsing failed: {error_message_ast}")

    if pred_dict_list is None:
        try:
            pred_dict_list = manual_ast_parsing(txt=txt)
        except Exception as e:
            error_message_ast = str(e)
            print(f"manual AST parsing failed: {error_message_ast}")

    if pred_dict_list is None:
        pred_dict_list = manual_xml_parsing(txt=txt)

    return pred_dict_list


def get_dict_list_from_tool_calls(
    tool_calls: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    has_parsing_error = False
    dict_list: List[Dict[str, Any]] = []

    try:
        for tool_call in tool_calls:
            if "function" in tool_call:
                func_obj = tool_call["function"]
                name = deepcopy(func_obj["name"]) if "name" in func_obj else ""
                arguments = (
                    deepcopy(func_obj["arguments"]) if "arguments" in func_obj else {}
                )
                if isinstance(arguments, str):
                    arguments = get_json_data_with_two_step_parsing(  # type: ignore
                        txt=arguments, should_return_list=False
                    )
                    obj = {"name": name, "arguments": arguments}
                    if "label" in func_obj and func_obj["label"] is not None:
                        obj["label"] = deepcopy(func_obj["label"])
                    elif "label" in tool_call and tool_call["label"] is not None:
                        obj["label"] = deepcopy(tool_call["label"])
                    # elif "id" in tool_call and tool_call["id"] is not None:
                    #     obj["label"] = deepcopy(tool_call["id"])
                dict_list.append({"name": name, "arguments": arguments})
    except Exception as e:
        print(e)
        has_parsing_error = True

    return dict_list, has_parsing_error


def parse_generated_text(
    prediction: EvaluationOutputResponseDataUnit, skip_grounding: bool
) -> Tuple[List[Dict[str, Any]], List[str], int, bool, List[str]]:
    generated_txt = prediction.generated_text
    pred_dict_list: List[Dict[str, Any]] = []
    parsing_error_messages: List[str] = []
    pred_func_calls = []
    pred_has_parsing_errors = False
    num_errors_parsing_pred_intent = 0
    try:
        if prediction.tool_calls is not None:
            pred_dict_list, has_parsing_error_instance = get_dict_list_from_tool_calls(
                tool_calls=prediction.tool_calls
            )
            if has_parsing_error_instance:
                raise Exception("tool call parsing error")
        else:
            pred_dict_list = parse_multi_step(txt=deepcopy(generated_txt))

        if skip_grounding:
            pred_func_calls = [json.dumps(func) for func in pred_dict_list]
        else:
            pred_func_calls_tmp = (
                ground_seq_nested_repsonse(pred_dict_list)
                if "label" in generated_txt
                else pred_dict_list
            )
            pred_func_calls = [json.dumps(func) for func in pred_func_calls_tmp]
    except Exception as e:
        print(e)
        parsing_error_messages.append(
            CommonErrorModel(error=str(e), payload=generated_txt).model_dump_json()
        )
        num_errors_parsing_pred_intent += 1
        pred_has_parsing_errors = True

    return (
        pred_dict_list,
        pred_func_calls,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
        parsing_error_messages,
    )


def parse_general_large_language_model_output(
    prediction: EvaluationOutputResponseDataUnit,
    num_errors_parsing_pred_intent: int,
    skip_grounding: bool = False,
) -> Tuple[
    List[str],
    List[str],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    int,
    bool,
    List[str],
]:
    pred_has_parsing_errors = False
    pred_func_calls: List[str] = []
    gold_func_calls: List[str] = []
    pred_dict_list: List[Dict[str, Any]] = []  # type: ignore
    parsing_error_messages: List[str] = []
    gold_dict_list = get_output_list(prediction=prediction)

    if skip_grounding:
        gold_func_calls = [json.dumps(func) for func in gold_dict_list]
    else:
        gold_func_calls_tmp = ground_seq_nested_repsonse(gold_dict_list)
        gold_func_calls = [json.dumps(func) for func in gold_func_calls_tmp]

    (
        pred_dict_list,
        pred_func_calls,
        num_errors_parsing_pred_intent_instance,
        pred_has_parsing_errors_instance,
        parsing_error_messages_instance,
    ) = parse_generated_text(prediction=prediction, skip_grounding=skip_grounding)
    parsing_error_messages.extend(parsing_error_messages_instance)
    pred_has_parsing_errors = (
        pred_has_parsing_errors or pred_has_parsing_errors_instance
    )
    num_errors_parsing_pred_intent += num_errors_parsing_pred_intent_instance

    return (
        pred_func_calls,
        gold_func_calls,
        (pred_dict_list if pred_dict_list is not None else []),
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
        parsing_error_messages,
    )


def parse_output_from_language_models_rest(
    prediction: EvaluationOutputResponseDataUnit,
    model_name: str,
    is_single_intent_detection: bool = False,
    is_agent: bool = False,
):
    model_name_lower_cased = model_name.lower()
    # hard code if u are running an agent, there is some assumption that is unclear
    # is_agent = True
    if is_agent:
        return parse_agent_rest(
            prediction,
            num_errors_parsing_pred_intent=0,
            is_single_intent_detection=is_single_intent_detection,
            skip_grounding=True,
        )
    if "llama" in model_name_lower_cased or "watt" in model_name_lower_cased:
        return parse_llama_3_70b_instruct_rest(
            prediction,
            num_errors_parsing_pred_intent=0,
            is_single_intent_detection=True,
            skip_grounding=True,
        )
    elif "mixtral" in model_name_lower_cased:
        return parse_mixtral_output_rest(
            prediction, num_errors_parsing_pred_intent=0, skip_grounding=True
        )
    elif (
        "gpt" in model_name_lower_cased
        or "deepseek" in model_name_lower_cased
        or "hammer" in model_name_lower_cased
        or "qwen" in model_name_lower_cased
    ):
        if "deepseek" in model_name_lower_cased:
            return parse_deepseek_output_rest(
                prediction,
                num_errors_parsing_pred_intent=0,
                skip_grounding=True,
                model_name="deepseek",
            )
        elif "hammer" in model_name_lower_cased:
            return parse_deepseek_output_rest(
                prediction,
                num_errors_parsing_pred_intent=0,
                skip_grounding=True,
                model_name="hammer",
            )
        elif "qwen" in model_name_lower_cased:
            return parse_deepseek_output_rest(
                prediction,
                num_errors_parsing_pred_intent=0,
                skip_grounding=True,
                model_name="qwen",
            )
        elif "gpt" in model_name_lower_cased:
            return parse_deepseek_output_rest(
                prediction,
                num_errors_parsing_pred_intent=0,
                skip_grounding=True,
                model_name="gpt",
            )
    else:
        return parse_llm_out_rest_dataset(
            prediction,
            num_errors_parsing_pred_intent=0,
            is_single_intent_detection=True,
            skip_grounding=True,
        )


def parse_output_from_language_models(
    prediction: EvaluationOutputResponseDataUnit,
    model_name: str,
    is_single_intent_detection: bool = False,
) -> Tuple[
    List[str],
    List[str],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    int,
    bool,
    List[str],
]:
    num_errors_parsing_pred_intent: int = 0
    pred_has_parsing_errors: bool = False
    pred_func_calls: List[str] = []
    gold_func_calls: List[str] = []
    pred_dict_list: List[Dict[str, Any]] = []
    gold_dict_list: List[Dict[str, Any]] = []
    parsing_error_messages: List[str] = []
    num_errors_parsing_pred_intent_res: int = 0
    if is_single_intent_detection:
        if prediction.num_preciedtion_parsing_errors is not None:  # use existing data
            pred_func_calls = prediction.predicted_function_calls
            gold_func_calls = prediction.gold_function_calls
            num_errors_parsing_pred_intent_res = (
                prediction.num_preciedtion_parsing_errors
            )
            pred_has_parsing_errors = num_errors_parsing_pred_intent_res > 0
        else:
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent_res,
                pred_has_parsing_errors,
                parsing_error_messages,
            ) = parse_output_from_language_models_rest(
                prediction, model_name, is_single_intent_detection, False
            )

    else:
        if prediction.num_preciedtion_parsing_errors is not None:  # use existing data
            pred_func_calls = prediction.predicted_function_calls
            gold_func_calls = prediction.gold_function_calls
            num_errors_parsing_pred_intent_res = (
                prediction.num_preciedtion_parsing_errors
            )
            pred_has_parsing_errors = num_errors_parsing_pred_intent_res > 0
        else:
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent_res,
                pred_has_parsing_errors,
                parsing_error_messages,
            ) = parse_general_large_language_model_output(
                prediction=prediction,
                num_errors_parsing_pred_intent=num_errors_parsing_pred_intent,
                skip_grounding=is_single_intent_detection,
            )
    # TODO: This is not correct for REST - Commenting it out
    # if is_single_intent_detection:
    #     if len(pred_func_calls) > 0:
    #         pred_func_calls = [pred_func_calls[0]]
    #     if len(pred_dict_list) > 0:
    #         pred_dict_list = [pred_dict_list[0]]

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent_res,
        pred_has_parsing_errors,
        parsing_error_messages,
    )
