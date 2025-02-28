from copy import deepcopy
import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, cast

from api_integrated_llm.data_models.common_models import CommonErrorModel
from api_integrated_llm.data_models.scorer_models import (
    ConfusionMatrixMode,
    ConfusionMetrixMetricsModel,
    MicroConfusionMetrixMetricsModel,
    ScorerOuputModel,
)
from api_integrated_llm.data_models.source_models import (
    EvaluationOutputResponseDataUnit,
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
from api_integrated_llm.helpers.metrics_helper import (
    get_confision_matrix_from_answers,
)
from api_integrated_llm.helpers.file_helper import (
    get_base_models_from_jsonl,
    get_dataset_name_from_file_path,
    get_uuid4_str,
    write_json,
    write_jsonl,
)


project_root_path = Path(__file__).parent.resolve()


def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t


def parse_output_from_language_models(
    prediction: Dict[str, Any],
    model_name: str,
    is_single_intent_detection: bool = False,
    is_agent: bool = False,
) -> Tuple[List[Any], List[Any], List[Any], List[Any], Any, Any, List[str]]:
    num_errors_parsing_pred_intent = 0
    pred_has_parsing_errors = False
    pred_func_calls, gold_func_calls = [], []
    pred_dict_list, gold_dict_list = [], []
    parsing_error_messages: List[str] = []
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
            parsing_error_messages,
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
                parsing_error_messages,
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
                parsing_error_messages,
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
                parsing_error_messages,
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
                parsing_error_messages,
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
            parsing_error_messages,
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
            parsing_error_messages,
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
            parsing_error_messages,
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
        parsing_error_messages,
    )


def get_function_dict(content: Any) -> Optional[Dict[str, Any]]:
    if content is None or not isinstance(content, str):
        return None
    # legacy: only for pred
    # if f.strip() == '{"name": "dummy", "arguments": {}}':
    #     continue
    return json.loads(
        content.replace("<|endoftext|>", "").replace("null", "{}").strip()
    )


def get_function_obj(
    content: Any, field_to_extract: Optional[str]
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    obj: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    try:
        obj = get_function_dict(content=content)
    except Exception as e:
        error_message = str(e)

    if obj is None:
        error_message = "function content is not str"
    elif field_to_extract not in obj:
        obj = None
        error_message = f"{field_to_extract} does not exist in a parsed object"

    return obj, error_message


def get_default_obj(field_to_extract: str) -> Union[str, Dict[str, Any]]:
    return get_uuid4_str() if field_to_extract == "name" else {}


def get_api_field_value(
    content: Any, field_to_extract: str
) -> Tuple[Union[str, Dict[str, Any]], int, bool]:
    obj, error_message = get_function_obj(
        content=content, field_to_extract=field_to_extract
    )
    obj_extracted = ""

    if obj is not None:
        obj_extracted = (
            obj[field_to_extract]
            if (error_message is None and field_to_extract in obj)
            else get_default_obj(field_to_extract=field_to_extract)
        )

    if field_to_extract == "name":
        obj_extracted = str(obj_extracted)

    num_errors = 1 if obj is None else 0
    has_parsing_errors = obj is None

    return obj_extracted, num_errors, has_parsing_errors


def get_api_contents(
    func_calls: List[Any], field_to_extract: str
) -> Tuple[List[Union[str, Dict[str, Any]]], int, bool]:
    apis_contents: List[Union[str, Dict[str, Any]]] = []
    has_parsing_errors_contents = False
    num_errors_contents = 0

    for content in func_calls:
        (obj_extracted, num_errors, has_parsing_errors) = get_api_field_value(
            content=content, field_to_extract=field_to_extract
        )
        apis_contents.append(obj_extracted)
        num_errors_contents += num_errors
        has_parsing_errors_contents = has_parsing_errors_contents or has_parsing_errors

    return (
        apis_contents,
        num_errors_contents,
        has_parsing_errors_contents,
    )


def get_api_lists_from_func_calls(
    func_calls: List[Any],
) -> Tuple[List[Tuple[str, List[str]]], int, bool]:
    api_list: List[Tuple[str, List[str]]] = []
    error_messages: List[str] = []
    num_errors_parsing_slot = 0
    has_parsing_errors_slot = False

    for content in func_calls:
        name_extracted, num_errors_name, has_no_name = get_api_field_value(
            content=content, field_to_extract="name"
        )

        if has_no_name or not isinstance(name_extracted, str):
            num_errors_parsing_slot += num_errors_name
            has_parsing_errors_slot = has_parsing_errors_slot or has_no_name
            error_messages.append("predicted function has no name")
            continue

        (
            arguments_extracted,
            num_errors_arguments,
            has_no_arguments,
        ) = get_api_field_value(content=content, field_to_extract="arguments")

        if (not has_no_arguments) and isinstance(arguments_extracted, dict):
            arguments: List[str] = []
            for arg, val in arguments_extracted.items():
                argument_str = f"{arg} = {val}"
                arguments.append(argument_str)
            api_list.append((name_extracted, arguments))
        else:
            api_list.append((name_extracted, []))
            num_errors_parsing_slot += num_errors_arguments
            has_parsing_errors_slot = has_parsing_errors_slot or has_no_arguments
            error_messages.append("predicted function has no valid arguments")

    return (
        api_list,
        num_errors_parsing_slot,
        has_parsing_errors_slot,
    )


def get_slot_info(
    gold_func_calls: List[Any], pred_func_calls: List[Any], intents_only: bool
) -> Tuple[List[List[str]], List[List[str]], List[str], int, int, bool, bool]:
    gold_output_slot: List[List[str]] = []
    pred_output_slot: List[List[str]] = []
    error_messages: List[str] = []
    num_errors_parsing_gold_slot = 0
    num_errors_parsing_pred_slot = 0
    pred_has_parsing_errors = False
    gold_has_parsing_errors = False

    if not intents_only:
        (
            pred_api_lists,
            instance_num_errors_parsing_slot,
            instance_has_parsing_errors_slot,
        ) = get_api_lists_from_func_calls(
            func_calls=pred_func_calls,
        )

        num_errors_parsing_pred_slot += instance_num_errors_parsing_slot
        pred_has_parsing_errors = (
            pred_has_parsing_errors or instance_has_parsing_errors_slot
        )

        (
            gold_api_lists,
            instance_num_errors_parsing_slot,
            instance_has_parsing_errors_slot,
        ) = get_api_lists_from_func_calls(
            func_calls=gold_func_calls,
        )
        num_errors_parsing_gold_slot += instance_num_errors_parsing_slot
        gold_has_parsing_errors = (
            gold_has_parsing_errors or instance_has_parsing_errors_slot
        )

        pred_api_dict = {api_name: arguments for api_name, arguments in pred_api_lists}

        for gold_api_name, gold_arguments in gold_api_lists:
            if gold_api_name in pred_api_dict:
                pred_output_slot.append(pred_api_dict[gold_api_name])
                gold_output_slot.append(deepcopy(gold_arguments))
            # Do not panaliize twice (once for API names and once for slots)
            # when predicted api_name does not exist

    return (
        gold_output_slot,
        pred_output_slot,
        error_messages,
        num_errors_parsing_gold_slot,
        num_errors_parsing_pred_slot,
        pred_has_parsing_errors,
        gold_has_parsing_errors,
    )


def get_item_metrics(
    predictions_input: List[EvaluationOutputResponseDataUnit],
    model_name: str,
    dataset_name: str,
    is_single_intent_detection: bool,
    intents_only: bool,
    win_rate_flag: bool,
    spec_path: Path,
) -> Tuple[
    List[List[str]],
    List[List[str]],
    List[List[str]],
    List[List[str]],
    int,
    int,
    int,
    int,
    int,
    List[float],
    int,
    int,
    List[str],
    List[str],
]:
    gold_output_intent: List[List[str]] = []
    pred_output_intent: List[List[str]] = []
    gold_output_slot: List[List[str]] = []
    pred_output_slot: List[List[str]] = []
    num_errors_parsing_pred_intent = 0
    num_errors_parsing_gold_intent = 0
    num_errors_parsing_pred_slot = 0
    num_errors_parsing_gold_slot = 0
    all_num_times_full_score = 0
    win_rate_list: List[float] = []
    num_pred_examples_w_parsing_errors = 0
    num_gold_examples_w_parsing_errors = 0
    error_messages: List[str] = []
    parsing_error_messages: List[str] = []

    for prediction, prediction_model in list(
        map(
            lambda item: (item.model_dump(), item.model_copy(deep=True)),
            predictions_input,
        )
    ):
        gold_has_parsing_errors = False
        pred_has_parsing_errors = False
        try:
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list,
                gold_dict_list,
                model_num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
                instance_parsing_error_messages,
            ) = parse_output_from_language_models(
                prediction=prediction,
                model_name=model_name[:],
                is_single_intent_detection=is_single_intent_detection,
                is_agent=prediction_model.is_agent,
            )
            num_errors_parsing_pred_intent += model_num_errors_parsing_pred_intent
            parsing_error_messages.extend(instance_parsing_error_messages)
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
            instance_num_errors_parsing_gold_intent,
            instance_gold_has_parsing_errors,
        ) = get_api_contents(
            func_calls=gold_func_calls,
            field_to_extract="name",
        )

        (
            pred_apis_names,
            instance_num_errors_parsing_pred_intent,
            instance_pred_has_parsing_errors,
        ) = get_api_contents(
            func_calls=pred_func_calls,
            field_to_extract="name",
        )

        pred_has_parsing_errors = (
            pred_has_parsing_errors or instance_pred_has_parsing_errors
        )
        gold_has_parsing_errors = (
            gold_has_parsing_errors or instance_gold_has_parsing_errors
        )
        num_errors_parsing_gold_intent += instance_num_errors_parsing_gold_intent
        num_errors_parsing_pred_intent += instance_num_errors_parsing_pred_intent

        gold_output_intent.append(cast(List[str], gold_apis_names))
        pred_output_intent.append(cast(List[str], pred_apis_names))

        (
            instance_gold_output_slot,
            instance_pred_output_slot,
            slot_error_messages,
            instance_num_errors_parsing_gold_slot,
            instance_num_errors_parsing_pred_slot,
            has_parsing_errors_pred,
            has_parsing_errors_gold,
        ) = get_slot_info(
            gold_func_calls=gold_func_calls,
            pred_func_calls=pred_func_calls,
            intents_only=intents_only,
        )
        gold_output_slot.extend(cast(List[List[str]], instance_gold_output_slot))
        pred_output_slot.extend(cast(List[List[str]], instance_pred_output_slot))
        error_messages.extend(slot_error_messages)
        num_errors_parsing_gold_slot += instance_num_errors_parsing_gold_slot
        num_errors_parsing_pred_slot += instance_num_errors_parsing_pred_slot
        pred_has_parsing_errors = pred_has_parsing_errors or has_parsing_errors_pred
        gold_has_parsing_errors = gold_has_parsing_errors or has_parsing_errors_gold

        num_pred_examples_w_parsing_errors += 1 if pred_has_parsing_errors else 0
        num_gold_examples_w_parsing_errors += 1 if gold_has_parsing_errors else 0

        ## Calculate WinRate here
        # win_rate_list.append(win_score)

    return (
        gold_output_intent,
        pred_output_intent,
        gold_output_slot,
        pred_output_slot,
        num_errors_parsing_pred_intent,
        num_errors_parsing_gold_intent,
        num_errors_parsing_pred_slot,
        num_errors_parsing_gold_slot,
        all_num_times_full_score,
        win_rate_list,
        num_pred_examples_w_parsing_errors,
        num_gold_examples_w_parsing_errors,
        error_messages,
        parsing_error_messages,
    )


def get_micro_confusion_matrix_metrics(
    gold_output_intent: List[List[str]],
    pred_output_intent: List[List[str]],
    gold_output_slot: List[List[str]],
    pred_output_slot: List[List[str]],
) -> MicroConfusionMetrixMetricsModel:
    return MicroConfusionMetrixMetricsModel(
        intent_set_metrics=ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
            get_confision_matrix_from_answers(
                gold_answers=gold_output_intent,
                predicted_answers=pred_output_intent,
                mode=ConfusionMatrixMode.SET,
            )
        ),
        intent_multiset_metrics=ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
            get_confision_matrix_from_answers(
                gold_answers=gold_output_intent,
                predicted_answers=pred_output_intent,
                mode=ConfusionMatrixMode.MULTISET,
            )
        ),
        intent_list_metrics=ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
            get_confision_matrix_from_answers(
                gold_answers=gold_output_intent,
                predicted_answers=pred_output_intent,
                mode=ConfusionMatrixMode.LIST,
            )
        ),
        slot_set_metrics=ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
            get_confision_matrix_from_answers(
                gold_answers=gold_output_slot,
                predicted_answers=pred_output_slot,
                mode=ConfusionMatrixMode.SET,
            )
        ),
    )


def calculate_scores(
    predictions_input: List[EvaluationOutputResponseDataUnit],
    intents_only: bool = False,
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
        all_num_times_full_score,
        win_rate_list,
        num_pred_examples_w_parsing_errors,
        num_gold_examples_w_parsing_errors,
        error_messages,
        parsing_error_messages,
    ) = get_item_metrics(
        predictions_input=predictions_input,
        model_name=model_name,
        dataset_name=dataset_name,
        is_single_intent_detection=is_single_intent_detection,
        intents_only=intents_only,
        win_rate_flag=win_rate_flag,
        spec_path=spec_path,
    )

    confusion_metrix_matrics_micro_model = get_micro_confusion_matrix_metrics(
        gold_output_intent=gold_output_intent,
        pred_output_intent=pred_output_intent,
        gold_output_slot=gold_output_slot,
        pred_output_slot=pred_output_slot,
    )

    num_samples = len(predictions_input)

    return ScorerOuputModel(
        confusion_metrix_matrics_micro=confusion_metrix_matrics_micro_model,
        num_examples=num_samples,
        percentage_times_full_score=(all_num_times_full_score / num_samples),
        win_rate=((sum(win_rate_list) / len(win_rate_list)) if win_rate_flag else None),
        num_errors_parsing_pred_intent=num_errors_parsing_pred_intent,
        num_errors_parsing_gold_intent=num_errors_parsing_gold_intent,
        num_errors_parsing_pred_slot=num_errors_parsing_pred_slot,
        num_errors_parsing_gold_slot=num_errors_parsing_gold_slot,
        num_pred_examples_w_parsing_errors=num_pred_examples_w_parsing_errors,
        num_gold_examples_w_parsing_errors=num_gold_examples_w_parsing_errors,
        error_messages=error_messages,
        parsing_error_messages=parsing_error_messages,
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
    win_rate_flag: bool = True,
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
