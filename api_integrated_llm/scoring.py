import os
import json
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional, Tuple
import sys
import importlib
import signal
from sklearn.metrics import accuracy_score

from api_integrated_llm.data_models.common_models import CommonErrorModel
from api_integrated_llm.data_models.source_models import (
    EvaluationOutputResponseDataUnit,
    ScorerOuputModel,
)
from api_integrated_llm.helpers.output_parsers import (
    parse_granite_20b_function_calling_output,
    parse_granite_3_output,
    parse_llama_3_70b_instruct,
    parse_llama_3_output,
    parse_mistral_7b_instruct_v0_3,
)
from api_integrated_llm.helpers.scorer_helper import (
    get_evaluation_output_response_data_units_from_json,
)
from api_integrated_llm.helpers.utils import (
    post_process_api_with_args,
)
from api_integrated_llm.helpers.metrics_helper import (
    compute_score,
    compute_score_sklearn,
)
from api_integrated_llm.helpers.file_helper import (
    get_base_models_from_jsonl,
    get_dataset_name_from_file_path,
    get_dict_from_json,
    write_json,
    write_jsonl,
)


project_root_path = Path(__file__).parent.resolve()


# Define a handler for the timeout
def handler(signum, frame):
    raise TimeoutError("Time limit exceeded!")


def calculate_ans_mathqa(func_calls, spec_lib):
    # Set the signal handler and a 10-second alarm
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(10)  # Set timeout for 10 seconds

    try:
        variable_result_map = {}
        for idx, f in enumerate(func_calls):
            label = f["label"].replace("$", "")
            # label = f["name"]
            output_params = [s for s in spec_lib if s["name"] == f["name"]][0][
                "output_parameter"
            ]
            output_params = list(output_params.keys())
            arg_values = []
            for k, v in f["arguments"].items():
                if type(v) == str and v.startswith("$") and v.endswith("$"):
                    v = v[1:-1]
                    v_l = v.split(".", 1)[0]
                    out_param = v.split(".", 1)[1]
                    v = variable_result_map[v_l][out_param]
                arg_values.append(str(v))

            func_str = f"{f['name']}({','.join(arg_values)})"
            res = eval(func_str)
            if (
                not type(res) == dict
                and not type(res) == list
                and len(output_params) == 1
            ):
                variable_result_map[label] = {output_params[0]: res}
            else:
                return False

        final_var = func_calls[-1]["label"].replace("$", "")

        final_ans = next(iter(variable_result_map[final_var].values()))

        return final_ans
    except TimeoutError:
        print("The program timed out!")
        signal.alarm(0)
        return False
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        return False


def calculate_ans_stack(func_calls, spec_lib, python_codes_dir, func_file_dict):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(10)
    try:
        variable_result_map = {}
        for idx, f in enumerate(func_calls):
            label = f["label"].replace("$", "")
            output_params = [s for s in spec_lib if s["name"] == f["name"]][0][
                "output_parameters"
            ][
                "properties"
            ]  # stack
            output_params = list(output_params.keys())
            arg_values = []
            arg_val_list = []

            for k, v in f["arguments"].items():
                if type(v) == str and v.startswith("$") and v.endswith("$"):
                    v = v[1:-1]
                    v_l = v.split(".", 1)[0]
                    out_param = v.split(".", 1)[1]
                    v = variable_result_map[v_l][out_param]
                arg_val_list.append(v)
                arg_values.append(str(v))

            ### stack
            file_name = func_file_dict[f["name"]]
            file_path = os.path.join(python_codes_dir, file_name)

            spec = importlib.util.spec_from_file_location(file_name, file_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[file_name] = mod
            spec.loader.exec_module(mod)
            func = getattr(mod, f["name"])
            res = func(*arg_val_list)

            if len(output_params) == 1:
                variable_result_map[label] = {output_params[0]: res}
            else:
                return False

        final_var = func_calls[-1]["label"].replace("$", "")
        final_ans = next(iter(variable_result_map[final_var].values()))

        return final_ans
    except TimeoutError:
        print("The program timed out!")
        signal.alarm(0)
        return False
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
        print(e)
        return False


def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t


def is_inner_sourced_json(obj: Any) -> bool:
    return not isinstance(obj, list)


def calculate_win_score(pred_func_calls, gold_ans, spec_file, dataset_name):
    if not pred_func_calls:
        return False
    spec = get_dict_from_json(spec_file)

    if is_inner_sourced_json(obj=spec):
        spec = list(spec["global_api_pool"].values())

    if dataset_name == "mathqa":
        pred_ans = calculate_ans_mathqa(pred_func_calls, spec)
    elif dataset_name == "stack":
        python_codes_dir = os.path.join(
            project_root_path,
            "data/py_code_files",
        )
        func_name_file_dict = get_dict_from_json(
            os.path.join(python_codes_dir, "func_file_map.json")
        )
        pred_ans = calculate_ans_stack(
            pred_func_calls, spec, python_codes_dir, func_name_file_dict
        )
    else:
        raise Exception("Dataset not handled")

    if type(gold_ans) == float and type(pred_ans) == float:
        dec_no = len(str(gold_ans).split(".")[1])
        pred_ans = round(pred_ans, dec_no)
    if pred_ans == gold_ans:
        return True
    else:
        if type(pred_ans) == tuple and type(gold_ans) == list:
            pred_ans = listit(pred_ans)
            if pred_ans == gold_ans:
                return True
        return False


def parse_output_from_language_models(
    prediction: Dict[str, Any],
    model_name: str,
    is_single_intent_detection: bool = False,
    is_agent: bool = False,
) -> Tuple[List[Any], List[Any], List[Any], List[Any], Any, Any]:
    num_errors_parsing_pred_intent = 0
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    num_errors_parsing_pred_intent_res = 0
    model_name_lower_cased = model_name.lower()
    if is_agent:
        (
            pred_func_calls,
            gold_func_calls,
            pred_dict_list,
            gold_dict_list,
            num_errors_parsing_pred_intent_res,
            pred_has_parsing_errors,
        ) = parse_llama_3_output(
            prediction=prediction,
            num_errors_parsing_pred_intent=num_errors_parsing_pred_intent,
            is_single_intent_detection=is_single_intent_detection,
            skip_grounding=is_single_intent_detection,
        )
    elif "granite" in model_name_lower_cased:
        if "functioncalling" in model_name_lower_cased:
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent_res,
                pred_has_parsing_errors,
            ) = parse_granite_20b_function_calling_output(
                prediction=prediction,
                num_errors_parsing_pred_intent=num_errors_parsing_pred_intent,
                is_single_intent_detection=is_single_intent_detection,
                skip_grounding=is_single_intent_detection,
            )
        else:
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent_res,
                pred_has_parsing_errors,
            ) = parse_granite_3_output(
                prediction=prediction,
                num_errors_parsing_pred_intent=num_errors_parsing_pred_intent,
                is_single_intent_detection=is_single_intent_detection,
                skip_grounding=is_single_intent_detection,
            )
    elif "llama" in model_name_lower_cased:
        if "llama-3-70b" in model_name_lower_cased:
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent_res,
                pred_has_parsing_errors,
            ) = parse_llama_3_70b_instruct(
                prediction=prediction,
                num_errors_parsing_pred_intent=num_errors_parsing_pred_intent,
                is_single_intent_detection=is_single_intent_detection,
                skip_grounding=is_single_intent_detection,
            )
        else:
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent_res,
                pred_has_parsing_errors,
            ) = parse_llama_3_output(
                prediction=prediction,
                num_errors_parsing_pred_intent=num_errors_parsing_pred_intent,
                is_single_intent_detection=is_single_intent_detection,
                skip_grounding=is_single_intent_detection,
            )
    elif "mistral" in model_name_lower_cased or "mixtral" in model_name_lower_cased:
        (
            pred_func_calls,
            gold_func_calls,
            pred_dict_list,
            gold_dict_list,
            num_errors_parsing_pred_intent_res,
            pred_has_parsing_errors,
        ) = parse_mistral_7b_instruct_v0_3(
            prediction=prediction,
            num_errors_parsing_pred_intent=num_errors_parsing_pred_intent,
            is_single_intent_detection=is_single_intent_detection,
            skip_grounding=is_single_intent_detection,
        )
    elif "deepseek" in model_name_lower_cased:
        (
            pred_func_calls,
            gold_func_calls,
            pred_dict_list,
            gold_dict_list,
            num_errors_parsing_pred_intent_res,
            pred_has_parsing_errors,
        ) = parse_llama_3_output(
            prediction=prediction,
            num_errors_parsing_pred_intent=num_errors_parsing_pred_intent,
            is_single_intent_detection=is_single_intent_detection,
            skip_grounding=is_single_intent_detection,
        )
    else:
        (
            pred_func_calls,
            gold_func_calls,
            pred_dict_list,
            gold_dict_list,
            num_errors_parsing_pred_intent_res,
            pred_has_parsing_errors,
        ) = parse_llama_3_output(
            prediction=prediction,
            num_errors_parsing_pred_intent=num_errors_parsing_pred_intent,
            is_single_intent_detection=is_single_intent_detection,
            skip_grounding=is_single_intent_detection,
        )

    return (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent_res,
        pred_has_parsing_errors,
    )


def get_api_names(
    gold_func_calls: List[Any], pred_func_calls: List[Any]
) -> Tuple[List[str], List[str], int, int, bool]:
    gold_apis_names, pred_apis_names = [], []
    pred_has_parsing_errors = False
    num_errors_parsing_pred_intent = 0
    num_errors_parsing_gold_intent = 0
    for f in pred_func_calls:
        if not f:
            continue
        try:
            if f.strip() == '{"name": "dummy", "arguments": {}}':
                continue
            f = json.loads(f.replace("<|endoftext|>", "").strip())
            pred_apis_names.append(str(f["name"]))
        except:
            # pred_apis_names.append('random_' + str(randrange(100)))
            num_errors_parsing_pred_intent += 1
            pred_has_parsing_errors = True
            pass
    for f in gold_func_calls:
        if not f:
            continue
        try:
            f = json.loads(f.replace("<|endoftext|>", "").replace("null", "{}").strip())
            gold_apis_names.append(str(f["name"]))
        except Exception as e:  # cases with empty gold output
            print(e)
            num_errors_parsing_gold_intent += 1
            pass

    return (
        gold_apis_names,
        pred_apis_names,
        num_errors_parsing_gold_intent,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
    )


def get_slot_info(
    gold_func_calls: List[Any], pred_func_calls: List[Any], intents_only: bool
) -> Tuple[List[Any], List[Any], List[str], int, int, bool]:
    gold_output_slot = []
    pred_output_slot = []
    error_messages: List[str] = []
    num_errors_parsing_gold_slot = 0
    num_errors_parsing_pred_slot = 0
    pred_has_parsing_errors = False

    if not intents_only:
        pred_api_map, gold_api_map = {}, {}  # type: ignore
        for f in pred_func_calls:
            if f.strip() == '{"name": "dummy", "arguments": {}}':
                continue
            try:
                if not f:
                    continue
                f = json.loads(f.replace("<|endoftext|>", "").strip())
                if type(f) != dict or "name" not in f:
                    raise Exception("'name' not in predicted function call")
                api_name = f["name"]
                pred_api_map[api_name] = []
                for arg, val in f["arguments"].items():
                    pred_api_map[f["name"]].append(f"{arg} = {val}")
            except Exception as e:
                num_errors_parsing_pred_slot += 1
                pred_has_parsing_errors = True
                error_messages.append(str(e))
                pass
        for f in gold_func_calls:
            if not f:
                continue
            try:
                f = json.loads(
                    f.replace("<|endoftext|>", "").replace("null", "{}").strip()
                )
                gold_api_map[f["name"]] = []
                for arg, val in f["arguments"].items():
                    gold_api_map[f["name"]].append(f"{arg} = {val}")
            except:  # cases with empty gold output
                num_errors_parsing_gold_slot += 1
                error_messages.append("gold output is empty")
                pass
        for key in set(pred_api_map.keys()).union(gold_api_map.keys()):
            if key in pred_api_map:
                pred_output_slot.append(pred_api_map[key])
            else:
                pred_output_slot.append([])
            if key in gold_api_map:
                gold_output_slot.append(gold_api_map[key])
            else:
                gold_output_slot.append([])

    return (
        gold_output_slot,
        pred_output_slot,
        error_messages,
        num_errors_parsing_gold_slot,
        num_errors_parsing_pred_slot,
        pred_has_parsing_errors,
    )


def get_api_args(
    gold_func_calls: List[Any], pred_func_calls: List[Any]
) -> Tuple[List[Any], List[Any]]:
    api_with_args_gold = []
    for f in gold_func_calls:
        f = json.loads(f.replace("<|endoftext|>", "").strip())
        f_name = str(f["name"])
        args = ", ".join(
            sorted([f"{key} = {val}" for key, val in f["arguments"].items()])
        )
        api_with_args_gold.append(f"{f_name}({args})")

    api_with_args_pred = []
    for f in pred_func_calls:
        try:
            f = json.loads(f.replace("<|endoftext|>", "").strip())
            f_name = str(f["name"])
            try:
                args = ", ".join(
                    sorted([f"{key} = {val}" for key, val in f["arguments"].items()])
                )
            except:
                args = {}  # type: ignore
            api_with_args_pred.append(f"{f_name}({args})")
        except:
            continue

    return post_process_api_with_args(api_with_args_gold, api_with_args_pred)


def get_accuracy_combined(
    api_with_args_gold: List[str], api_with_args_pred: List[str], spec_path: Path
) -> Tuple[float, Optional[str]]:
    error_message: Optional[str] = None
    accuracy_combined = 0.0
    try:
        accuracy_combined = accuracy_score(api_with_args_gold, api_with_args_pred)
    except Exception as e:
        print(e)
        error_message = CommonErrorModel(
            error="Error at accuracy_score(): " + str(e), file=str(spec_path)
        ).model_dump_json()

    return accuracy_combined, error_message


def get_item_metrics(
    predictions_input: List[EvaluationOutputResponseDataUnit],
    model_name: str,
    dataset_name: str,
    is_single_intent_detection: bool,
    intents_only: bool,
    win_rate_flag: bool,
    spec_path: Path,
) -> Tuple[
    List[Any],
    List[Any],
    List[Any],
    List[Any],
    int,
    int,
    int,
    int,
    List[float],
    int,
    List[float],
    int,
    List[str],
]:
    gold_output_intent = []
    pred_output_intent = []
    gold_output_slot = []
    pred_output_slot = []
    num_errors_parsing_pred_intent = 0
    num_errors_parsing_gold_intent = 0
    num_errors_parsing_pred_slot = 0
    num_errors_parsing_gold_slot = 0
    all_accuracy_combined = []
    all_num_times_full_score = 0
    win_rate_list = []
    num_pred_examples_w_parsing_errors = 0
    error_messages: List[str] = []

    for prediction, prediction_model in list(
        map(
            lambda item: (item.model_dump(), item.model_copy(deep=True)),
            predictions_input,
        )
    ):
        try:
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                model_num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
            ) = parse_output_from_language_models(
                prediction=prediction,
                model_name=model_name[:],
                is_single_intent_detection=is_single_intent_detection,
                is_agent=prediction_model.is_agent,
            )
            num_errors_parsing_pred_intent += model_num_errors_parsing_pred_intent
        except Exception as e:
            print(e)
            error_messages.append(
                CommonErrorModel(
                    error="Error at parse_output_from_language_models(): " + str(e),
                    file=str(spec_path),
                ).model_dump_json()
            )
            continue

        (
            gold_apis_names,
            pred_apis_names,
            instance_num_errors_parsing_gold_intent,
            instance_num_errors_parsing_pred_intent,
            has_parsing_errors,
        ) = get_api_names(
            gold_func_calls=gold_func_calls, pred_func_calls=pred_func_calls
        )
        pred_has_parsing_errors = pred_has_parsing_errors or has_parsing_errors
        num_errors_parsing_gold_intent += instance_num_errors_parsing_gold_intent
        num_errors_parsing_pred_intent += instance_num_errors_parsing_pred_intent

        gold_output_intent.append(gold_apis_names)
        pred_output_intent.append(pred_apis_names)

        (
            instance_gold_output_slot,
            instance_pred_output_slot,
            slot_error_messages,
            instance_num_errors_parsing_gold_slot,
            instance_num_errors_parsing_pred_slot,
            has_parsing_errors,
        ) = get_slot_info(
            gold_func_calls=gold_func_calls,
            pred_func_calls=pred_func_calls,
            intents_only=intents_only,
        )
        gold_output_slot.extend(instance_gold_output_slot)
        pred_output_slot.extend(instance_pred_output_slot)
        error_messages.extend(slot_error_messages)
        num_errors_parsing_gold_slot += instance_num_errors_parsing_gold_slot
        num_errors_parsing_pred_slot += instance_num_errors_parsing_pred_slot
        pred_has_parsing_errors = pred_has_parsing_errors or has_parsing_errors

        if pred_has_parsing_errors:
            num_pred_examples_w_parsing_errors += 1

        api_with_args_gold, api_with_args_pred = get_api_args(
            gold_func_calls=gold_func_calls, pred_func_calls=pred_func_calls
        )

        accuracy_combined, error_message_intance = get_accuracy_combined(
            api_with_args_gold=api_with_args_gold,
            api_with_args_pred=api_with_args_pred,
            spec_path=spec_path,
        )
        all_accuracy_combined.append(accuracy_combined)

        if accuracy_combined == 1:
            all_num_times_full_score += 1

        if error_message_intance is not None:
            error_messages.append(error_message_intance)

        ## WinRate
        if win_rate_flag:
            win_score = calculate_win_score(
                pred_dict_list, prediction["gold_answer"], spec_path, dataset_name
            )
            win_rate_list.append(win_score)
    return (
        gold_output_intent,
        pred_output_intent,
        gold_output_slot,
        pred_output_slot,
        num_errors_parsing_pred_intent,
        num_errors_parsing_gold_intent,
        num_errors_parsing_pred_slot,
        num_errors_parsing_gold_slot,
        all_accuracy_combined,
        all_num_times_full_score,
        win_rate_list,
        num_pred_examples_w_parsing_errors,
        error_messages,
    )


def get_metrics_confusion_matirx(
    sklearn_metrics: bool,
    intents_only: bool,
    gold_output_intent: List[Any],
    pred_output_intent: List[Any],
    gold_output_slot: List[Any],
    pred_output_slot: List[Any],
) -> Tuple[
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
    Optional[float],
]:
    p_intent, r_intent, f1_intent, p_slot, r_slot, f1_slot = (
        None,
        None,
        None,
        None,
        None,
        None,
    )

    if not sklearn_metrics:
        p_intent, r_intent, f1_intent = compute_score(
            gold_output_intent, pred_output_intent
        )
    else:
        p_intent, r_intent, f1_intent = compute_score_sklearn(
            gold_output_intent, pred_output_intent
        )

    if not intents_only:
        if not sklearn_metrics:
            p_slot, r_slot, f1_slot = compute_score(gold_output_slot, pred_output_slot)
        else:
            p_slot, r_slot, f1_slot = compute_score_sklearn(
                gold_output_slot, pred_output_slot
            )

    return (p_intent, r_intent, f1_intent, p_slot, r_slot, f1_slot)


def calculate_scores(
    predictions_input: List[EvaluationOutputResponseDataUnit],
    intents_only: bool = False,
    sklearn_metrics: bool = True,
    win_rate_flag: bool = True,
    is_single_intent_detection: bool = False,
) -> ScorerOuputModel:
    (
        model_name,
        dataset_name,
        spec_path,
        model_temperature,
        model_max_tokens,
    ) = predictions_input[0].get_dataset_basic_info()

    (
        gold_output_intent,
        pred_output_intent,
        gold_output_slot,
        pred_output_slot,
        num_errors_parsing_pred_intent,
        num_errors_parsing_gold_intent,
        num_errors_parsing_pred_slot,
        num_errors_parsing_gold_slot,
        all_accuracy_combined,
        all_num_times_full_score,
        win_rate_list,
        num_pred_examples_w_parsing_errors,
        error_messages,
    ) = get_item_metrics(
        predictions_input=predictions_input,
        model_name=model_name,
        dataset_name=dataset_name,
        is_single_intent_detection=is_single_intent_detection,
        intents_only=intents_only,
        win_rate_flag=win_rate_flag,
        spec_path=spec_path,
    )

    (
        p_intent,
        r_intent,
        f1_intent,
        p_slot,
        r_slot,
        f1_slot,
    ) = get_metrics_confusion_matirx(
        sklearn_metrics=sklearn_metrics,
        intents_only=intents_only,
        gold_output_intent=gold_output_intent,
        pred_output_intent=pred_output_intent,
        gold_output_slot=gold_output_slot,
        pred_output_slot=pred_output_slot,
    )

    num_samples = len(predictions_input)

    return ScorerOuputModel(
        p_intent=p_intent,
        r_intent=r_intent,
        f1_intent=f1_intent,
        p_slot=p_slot,
        r_slot=r_slot,
        f1_slot=f1_slot,
        num_examples=num_samples,
        accuracy_combined=statistics.mean(all_accuracy_combined),
        percentage_times_full_score=(all_num_times_full_score / num_samples),
        win_rate=((sum(win_rate_list) / len(win_rate_list)) if win_rate_flag else None),
        num_errors_parsing_pred_intent=num_errors_parsing_pred_intent,
        num_errors_parsing_gold_intent=num_errors_parsing_gold_intent,
        num_errors_parsing_pred_slot=num_errors_parsing_pred_slot,
        num_errors_parsing_gold_slot=num_errors_parsing_gold_slot,
        num_pred_examples_w_parsing_errors=num_pred_examples_w_parsing_errors,
        error_messages=error_messages,
        model_temperature=model_temperature,
        model_max_tokens=model_max_tokens,
        evaluation_source=list(
            map(lambda item: item.model_copy(deep=True), predictions_input)
        ),
        gold_output_intent=gold_output_intent,
        pred_output_intent=pred_output_intent,
        gold_output_slot=gold_output_slot,
        pred_output_slot=pred_output_slot,
    )


def handle_scoring_process_exception(
    output_root_path: Path,
    e: Exception,
    model_name: str,
    dataset_name: str,
    evaluator_output_file_path: Path,
    temperature_str: str,
    max_tokens_str: str,
) -> None:
    print(e)
    write_jsonl(
        file_path=Path(
            os.path.join(
                output_root_path,
                "error",
                model_name,
                temperature_str,
                max_tokens_str,
                dataset_name + "_scoring" + ".json",
            )
        ),
        jsons=[CommonErrorModel(error=str(e), file=str(evaluator_output_file_path))],
        should_append=True,
    )


def check_single_intent(evaluator_output_file_path: Path) -> bool:
    return "rest" in str(evaluator_output_file_path)


def scoring(
    evaluator_output_file_paths: List[Path],
    output_folder_path: Path,
    win_rate_flag: bool = False,
) -> None:
    for evaluator_output_file_path in evaluator_output_file_paths:
        dataset_name = get_dataset_name_from_file_path(
            file_path=evaluator_output_file_path
        )
        temperature_str = "default_temperature"
        max_tokens_str = "default_max_tokens"
        model_name = "default_model"
        try:
            data: List[EvaluationOutputResponseDataUnit] = (
                get_base_models_from_jsonl(
                    file_path=evaluator_output_file_path,
                    base_model=EvaluationOutputResponseDataUnit,
                )
                if str(evaluator_output_file_path).endswith("jsonl")
                else get_evaluation_output_response_data_units_from_json(
                    file_path=evaluator_output_file_path,
                )
            )

            if data is None or len(data) == 0:
                raise Exception(
                    f"No evaluation data found at {evaluator_output_file_path}"
                )

            temperature_str, max_tokens_str, dataset_name, model_name = data[
                0
            ].get_basic_strs()

            is_single_intent_detection = check_single_intent(
                evaluator_output_file_path=evaluator_output_file_path
            )

            write_json(
                file_path=Path(
                    os.path.join(
                        output_folder_path,
                        model_name,
                        temperature_str,
                        max_tokens_str,
                        (dataset_name + "_scoring_output.json"),
                    )
                ),
                base_model=calculate_scores(
                    data,
                    sklearn_metrics=(not is_single_intent_detection),
                    win_rate_flag=win_rate_flag,
                    is_single_intent_detection=is_single_intent_detection,
                ),
            )
        except Exception as e:
            handle_scoring_process_exception(
                output_root_path=output_folder_path,
                e=e,
                model_name=model_name,
                dataset_name=dataset_name,
                evaluator_output_file_path=evaluator_output_file_path,
                temperature_str=temperature_str,
                max_tokens_str=max_tokens_str,
            )
