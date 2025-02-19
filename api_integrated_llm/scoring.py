from copy import deepcopy
import os
import json
from pathlib import Path
import statistics
from typing import Any, Dict, List, Optional
from tqdm import tqdm
import sys

from api_integrated_llm.helpers.output_parsers import (
    parse_Hammer2_0_7b,
    parse_granite_20b_function_calling_output,
    parse_granite_3_output,
    parse_hermes_2_pro_mistral_7B,
    parse_llama_3_70b_instruct,
    parse_llama_3_output,
    parse_mistral_7b_instruct_v0_3,
    parse_xLAM_1b_fc_r,
)
from api_integrated_llm.helpers.utils import (
    post_process_api_with_args,
)
from api_integrated_llm.helpers.metrics_helper import (
    compute_score,
    compute_score_sklearn,
)
from api_integrated_llm.helpers.file_helper import (
    get_dict_from_json,
    get_list_dict_from_jsonl,
)
import importlib
import signal

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
                # ipdb.set_trace()
                return False
        # ipdb.set_trace()
        final_var = func_calls[-1]["label"].replace("$", "")
        # final_var = func_calls[-1]["name"].replace("$", "")
        final_ans = next(iter(variable_result_map[final_var].values()))
        # ipdb.set_trace()
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
        # ipdb.set_trace()
        return False


def calculate_ans_stack(func_calls, spec_lib, python_codes_dir, func_file_dict):
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(10)
    try:
        # ipdb.set_trace()
        variable_result_map = {}
        for idx, f in enumerate(func_calls):
            label = f["label"].replace("$", "")
            # output_params = [s for s in spec_lib if s["name"] == f["name"]][0]["output_parameter"] # mathqa
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
        # ipdb.set_trace()
        final_var = func_calls[-1]["label"].replace("$", "")
        final_ans = next(iter(variable_result_map[final_var].values()))
        # ipdb.set_trace()
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
        # ipdb.set_trace()
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


def calculate_scores(
    predictions,
    model_name,
    spec_path,
    dataset_name,
    intents_only=False,
    sklearn_metrics=True,
    win_rate_flag=True,
    model_temperature: float = 0.0,
    model_max_tokens: int = 1500,
):
    error_messages: List[str] = []
    # ipdb.set_trace()
    gold_output_intent = []
    pred_output_intent = []
    gold_output_slot = []
    pred_output_slot = []
    p_intent, r_intent, f1_intent, p_slot, r_slot, f1_slot = (
        None,
        None,
        None,
        None,
        None,
        None,
    )
    num_errors_parsing_pred_intent = 0
    num_errors_parsing_gold_intent = 0
    num_errors_parsing_pred_slot = 0
    num_errors_parsing_gold_slot = 0
    all_accuracy_combined = []
    all_num_times_full_score = 0
    win_rate_list = []
    num_pred_examples_w_parsing_errors = 0
    for item in tqdm(predictions):
        pred_has_parsing_errors = False
        pred_func_calls, gold_func_calls = [], []
        pred_dict_list, gold_dict_list = [], []
        if "granite" in model_name.lower() and "functioncalling" in model_name.lower():
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
            ) = parse_granite_20b_function_calling_output(
                item, num_errors_parsing_pred_intent
            )
        elif "llama-3" in model_name.lower():
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
            ) = parse_llama_3_70b_instruct(item, num_errors_parsing_pred_intent)
        elif "mistral" in model_name.lower() or "mixtral" in model_name.lower():
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
            ) = parse_mistral_7b_instruct_v0_3(item, num_errors_parsing_pred_intent)
        elif "hermes" in model_name.lower():
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
            ) = parse_hermes_2_pro_mistral_7B(item, num_errors_parsing_pred_intent)
        elif "xlam" in model_name.lower():
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
            ) = parse_xLAM_1b_fc_r(item, num_errors_parsing_pred_intent)
        elif "hammer" in model_name.lower():  # Todo: Hammer parsing @Mayank
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
            ) = parse_Hammer2_0_7b(item, num_errors_parsing_pred_intent)
        elif "granite" in model_name.lower():
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
            ) = parse_granite_3_output(item, num_errors_parsing_pred_intent)
        elif "llama" in model_name.lower():
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
            ) = parse_llama_3_output(item, num_errors_parsing_pred_intent)
        elif "deepseek" in model_name.lower():
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
            ) = parse_llama_3_output(item, num_errors_parsing_pred_intent)
        else:
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
            ) = parse_llama_3_output(item, num_errors_parsing_pred_intent)

        gold_apis_names, pred_apis_names = [], []
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
                f = json.loads(f.replace("<|endoftext|>", "").strip())
                gold_apis_names.append(str(f["name"]))
            except:  # cases with empty gold output
                num_errors_parsing_gold_intent += 1
                pass

        gold_output_intent.append(gold_apis_names)
        pred_output_intent.append(pred_apis_names)
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
                    f = json.loads(f.replace("<|endoftext|>", "").strip())
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

        if pred_has_parsing_errors:
            num_pred_examples_w_parsing_errors += 1

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
                        sorted(
                            [f"{key} = {val}" for key, val in f["arguments"].items()]
                        )
                    )
                except:
                    args = {}  # type: ignore
                api_with_args_pred.append(f"{f_name}({args})")
            except:
                continue

        api_with_args_gold, api_with_args_pred = post_process_api_with_args(
            api_with_args_gold, api_with_args_pred
        )

        from sklearn.metrics import accuracy_score

        try:
            accuracy_combined = accuracy_score(api_with_args_gold, api_with_args_pred)
        except:
            accuracy_combined = 0.0
        if accuracy_combined == 1:
            all_num_times_full_score += 1
        all_accuracy_combined.append(accuracy_combined)

        ## WinRate
        if win_rate_flag:
            # win_score = calculate_win_score(gold_dict_list, item["gold_answer"], spec_path)
            win_score = calculate_win_score(
                pred_dict_list, item["gold_answer"], spec_path, dataset_name
            )
            win_rate_list.append(win_score)

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

    return {
        "p_intent": "{:.3f}".format(p_intent),
        "r_intent": "{:.3f}".format(r_intent),
        "f1_intent": "{:.3f}".format(f1_intent),
        "p_slot": (
            "{:.3f}".format(p_slot)
            if p_slot is not None
            else ""  # noqa: E711 # type: ignore
        ),
        "r_slot": (
            "{:.3f}".format(r_slot)
            if r_slot is not None
            else ""  # noqa: E711 # type: ignore
        ),
        "f1_slot": (
            "{:.3f}".format(f1_slot)
            if f1_slot is not None
            else ""  # noqa: E711 # type: ignore
        ),
        "num_examples": len(predictions),
        "accuracy_combined": "{:.3f}".format(statistics.mean(all_accuracy_combined)),
        "percentage_times_full_score": "{:.3f}".format(
            all_num_times_full_score / len(predictions)
        ),
        "win_rate": (
            "{:.3f}".format(sum(win_rate_list) / len(win_rate_list))
            if win_rate_flag
            else "no"
        ),
        "num_errors_parsing_pred_intent": num_errors_parsing_pred_intent,
        "num_errors_parsing_gold_intent": num_errors_parsing_gold_intent,
        "num_errors_parsing_pred_slot": num_errors_parsing_pred_slot,
        "num_errors_parsing_gold_slot": num_errors_parsing_gold_slot,
        "num_pred_examples_w_parsing_errors": num_pred_examples_w_parsing_errors,
        "error_messages": error_messages,
        "model_temperature": model_temperature,
        "model_max_tokens": model_max_tokens,
        "evaluation_source": deepcopy(predictions),
        "gold_output_intent": gold_output_intent,
        "pred_output_intent": pred_output_intent,
        "gold_output_slot": gold_output_slot,
        "pred_output_slot": pred_output_slot,
    }


def print_result(result, model, dataset):
    print("\n###################################")
    print(f"############ {dataset} ##############")
    print(f"############ {model} ##############")
    print("###################################")
    print(f"Total Samples: {result['num_examples']}")
    print(f"Parsing Errors: {result['num_pred_examples_w_parsing_errors']}")
    print(f"F1 Intent: {result['f1_intent']}")
    print(f"F1 Slot: {result['f1_slot']}")
    print(f"Partial Match Accuracy: {result['accuracy_combined']}")
    print(f"Full Match Accuracy: {result['percentage_times_full_score']}")
    print(f"Win Rate: {result['win_rate']}")
    print("-" * 100)


def create_folders_recirsively_if_not_exist(tmp_path: str) -> None:
    base_path = os.path.basename(os.path.normpath(tmp_path))
    directory_path = (
        os.path.dirname(os.path.abspath(tmp_path))  # file path
        if "." in base_path
        else os.path.abspath(tmp_path)  # folder path
    )

    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def write_json_from_dict(file_path: str, dic: Dict) -> None:
    create_folders_recirsively_if_not_exist(tmp_path=file_path)

    with open(file_path, "w") as outfile:
        json.dump(dic, outfile)


def get_dataset_name_from_file_path(file_path: Path) -> str:
    file_name = str(file_path).split("/")[-1]
    return file_name.replace(".jsonl", "")


def get_files_in_folder(
    folder_path: str, file_extension: Optional[str] = None
) -> List[str]:
    return (
        [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(folder_path)
            for f in filenames
            if os.path.splitext(f)[1] == ("." + file_extension)
        ]
        if file_extension is not None
        else [
            os.path.join(dp, f)
            for dp, dn, filenames in os.walk(folder_path)
            for f in filenames
        ]
    )


def check_data(
    data: Optional[List[Dict[str, Any]]],
    dataset_name: str,
    evaluator_output_file_path: Path,
) -> bool:
    if data is None or len(data) == 0:  # handle empty data
        save_path = os.path.join(
            project_root_path,
            "error",
            dataset_name + "_scoring" + ".json",
        )
        write_json_from_dict(
            file_path=save_path,
            dic={
                "error": "No data to score",
                "file": evaluator_output_file_path,
            },
        )
        return False
    return True


def handle_scoring_process_exception(
    e: Exception,
    model_name: str,
    dataset_name: str,
    evaluator_output_file_path: Path,
    temperature_str: str,
    max_tokens_str: str,
) -> None:
    print(e)
    write_json_from_dict(
        file_path=os.path.join(
            project_root_path,
            "error",
            model_name,
            temperature_str,
            max_tokens_str,
            dataset_name + "_scoring" + ".json",
        ),
        dic={"error": str(e), "file": evaluator_output_file_path},
    )


def scoring(
    evaluator_output_file_paths: List[Path],
    output_folder_path: Path,
    win_rate_flag: bool = False,
) -> None:
    for evaluator_output_file_path in evaluator_output_file_paths:
        dataset_name = get_dataset_name_from_file_path(
            file_path=evaluator_output_file_path
        )

        try:
            data = get_list_dict_from_jsonl(evaluator_output_file_path)
            temperature_str = "temperature_" + str(data[0]["temperature"]).replace(
                ".", "_"
            )
            max_tokens_str = "maxtokens_" + str(data[0]["max_tokens"])

            if not check_data(
                data=data,
                dataset_name=dataset_name,
                evaluator_output_file_path=evaluator_output_file_path,
            ):
                continue

            dataset_name = data[0]["dataset_name"][:]
            model = data[0]["llm_model_id"]
            write_json_from_dict(
                file_path=os.path.join(
                    output_folder_path,
                    model,
                    temperature_str,
                    max_tokens_str,
                    (dataset_name + "_scoring_output.json"),
                ),
                dic=calculate_scores(
                    data,
                    model,
                    data[0]["source_file_path"],
                    dataset_name,
                    win_rate_flag=win_rate_flag,
                    model_temperature=data[0]["temperature"],
                    model_max_tokens=data[0]["max_tokens"],
                ),
            )
        except Exception as e:
            handle_scoring_process_exception(
                e=e,
                model_name=model,
                dataset_name=dataset_name,
                evaluator_output_file_path=evaluator_output_file_path,
                temperature_str=temperature_str,
                max_tokens_str=max_tokens_str,
            )
