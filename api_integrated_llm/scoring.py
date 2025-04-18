from collections import deque
from copy import deepcopy
import os
import json
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union, cast

from api_integrated_llm.data_models.auxiliary_models import SampleIgonoreModel
from api_integrated_llm.data_models.common_models import CommonErrorModel
from api_integrated_llm.data_models.scorer_models import (
    ConfusionMatrixMode,
    ConfusionMatrixModel,
    ConfusionMetrixMetricsModel,
    ContentPairModel,
    MicroConfusionMetrixMetricsByOutputLengthModel,
    MicroConfusionMetrixMetricsByOutputLengthProblemLevelModel,
    MicroConfusionMetrixMetricsModel,
    MicroConfusionMetrixMetricsProblemLevelModel,
    ScorerOuputModel,
)
from api_integrated_llm.data_models.source_models import (
    EvaluationOutputResponseDataUnit,
)
from api_integrated_llm.helpers.output_parsers import (
    parse_output_from_language_models,
    parse_output_from_language_models_rest,
)
from api_integrated_llm.helpers.scorer_helper import (
    get_evaluation_output_response_data_units,
    get_sample_ignore_model_from_file,
)
from api_integrated_llm.helpers.metrics_helper import (
    get_confision_matrix_from_answers,
    get_confision_matrix_from_answers_by_output_length,
    get_confision_matrix_from_answers_by_output_length_slot,
    get_confision_matrix_from_answers_slot,
)
from api_integrated_llm.helpers.file_helper import (
    get_uuid4_str,
    write_json,
    write_jsonl,
)


project_root_path = Path(__file__).parent.resolve()


def get_function_dict(content: Any) -> Optional[Dict[str, Any]]:
    if content is None or not isinstance(content, str):
        return None

    return json.loads(
        content.replace("<|endoftext|>", "").replace("null", "{}").strip()
    )


def clean_up_pred(pred):
    new_pred = []
    for p in pred:
        if p in (None, "{}", {}):
            p = ""
        new_pred.append(p)
    return new_pred


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
        try:
            obj_extracted = (
                obj[field_to_extract]
                if (error_message is None and field_to_extract in obj)
                else get_default_obj(field_to_extract=field_to_extract)
            )
        except:
            # todo: work around for function names that have "name" in them
            obj_extracted = ""
    if obj_extracted and field_to_extract == "name":
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
    for idx, content in enumerate(func_calls):
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


def get_api_dict_with_list_as_value(
    api_lists: List[Tuple[str, List[str]]],
) -> Dict[str, Deque[List[str]]]:
    api_dict: Dict[str, Deque[List[str]]] = {}
    for api_name, arguments in api_lists:
        if api_name not in api_dict:
            api_dict[api_name] = deque()
        api_dict[api_name].append(deepcopy(arguments))
    return api_dict


def get_slot_info(
    gold_func_calls: List[Any], pred_func_calls: List[Any]
) -> Tuple[List[List[str]], List[List[str]], List[str], int, int, bool, bool,]:
    gold_output_slot: List[List[str]] = []
    pred_output_slot: List[List[str]] = []
    error_messages: List[str] = []
    num_errors_parsing_gold_slot = 0
    num_errors_parsing_pred_slot = 0
    pred_has_parsing_errors = False
    gold_has_parsing_errors = False

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

    pred_api_dict: Dict[str, Deque[List[str]]] = get_api_dict_with_list_as_value(
        api_lists=pred_api_lists
    )

    for gold_api_name, gold_arguments in gold_api_lists:
        if gold_api_name in pred_api_dict:
            pred_arguments = pred_api_dict[gold_api_name].popleft()

            if len(pred_api_dict[gold_api_name]) == 0:
                pred_api_dict.pop(gold_api_name, "")

            pred_output_slot.append(deepcopy(pred_arguments))
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
    is_single_intent_detection: bool,
    spec_path: Path,
) -> Tuple[
    List[List[str]],
    List[List[str]],
    List[List[str]],
    List[List[str]],
    List[List[List[str]]],
    List[List[List[str]]],
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    List[str],
    List[str],
    List[Union[str, int]],
    List[List[str]],
    List[List[str]],
    List[List[Dict[str, Any]]],
    List[List[Dict[str, Any]]],
]:
    gold_output_intent: List[List[str]] = []
    pred_output_intent: List[List[str]] = []
    gold_output_slot: List[List[str]] = []
    pred_output_slot: List[List[str]] = []
    gold_output_slot_problem: List[List[List[str]]] = []
    pred_output_slot_problem: List[List[List[str]]] = []
    num_errors_parsing_pred_intent = 0
    num_errors_parsing_gold_intent = 0
    num_errors_parsing_pred_slot = 0
    num_errors_parsing_gold_slot = 0
    all_num_times_full_score = 0
    num_pred_examples_w_parsing_errors = 0
    num_gold_examples_w_parsing_errors = 0
    error_messages: List[str] = []
    parsing_error_messages: List[str] = []
    sample_ids: List[Union[str, int]] = []
    predicted_function_calls: List[List[str]] = []
    gold_function_calls: List[List[str]] = []
    pred_dict_list: List[List[Dict[str, Any]]] = []
    gold_dict_list: List[List[Dict[str, Any]]] = []

    for prediction_model in predictions_input:
        gold_has_parsing_errors = False
        pred_has_parsing_errors = False
        try:
            (
                pred_func_calls,
                gold_func_calls,
                pred_dict_list_instance,
                gold_dict_list_instance,
                model_num_errors_parsing_pred_intent,
                pred_has_parsing_errors,
                instance_parsing_error_messages,
            ) = parse_output_from_language_models(
                prediction=prediction_model,
                model_name=model_name[:],
                is_single_intent_detection=is_single_intent_detection,
            )
            num_errors_parsing_pred_intent += model_num_errors_parsing_pred_intent
            parsing_error_messages.extend(instance_parsing_error_messages)
            predicted_function_calls.append(pred_func_calls)
            gold_function_calls.append(gold_func_calls)
            pred_dict_list.append(pred_dict_list_instance)
            gold_dict_list.append(gold_dict_list_instance)
            sample_ids.append(prediction_model.sample_id)
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
        )
        gold_output_slot.extend(cast(List[List[str]], instance_gold_output_slot))
        pred_output_slot.extend(cast(List[List[str]], instance_pred_output_slot))
        gold_output_slot_problem.append(deepcopy(instance_gold_output_slot))
        pred_output_slot_problem.append(deepcopy(instance_pred_output_slot))
        error_messages.extend(slot_error_messages)
        num_errors_parsing_gold_slot += instance_num_errors_parsing_gold_slot
        num_errors_parsing_pred_slot += instance_num_errors_parsing_pred_slot
        pred_has_parsing_errors = pred_has_parsing_errors or has_parsing_errors_pred
        gold_has_parsing_errors = gold_has_parsing_errors or has_parsing_errors_gold

        num_pred_examples_w_parsing_errors += 1 if pred_has_parsing_errors else 0
        num_gold_examples_w_parsing_errors += 1 if gold_has_parsing_errors else 0

    return (
        gold_output_intent,
        pred_output_intent,
        gold_output_slot,
        pred_output_slot,
        gold_output_slot_problem,
        pred_output_slot_problem,
        num_errors_parsing_pred_intent,
        num_errors_parsing_gold_intent,
        num_errors_parsing_pred_slot,
        num_errors_parsing_gold_slot,
        all_num_times_full_score,
        num_pred_examples_w_parsing_errors,
        num_gold_examples_w_parsing_errors,
        error_messages,
        parsing_error_messages,
        sample_ids,
        predicted_function_calls,
        gold_function_calls,
        pred_dict_list,
        gold_dict_list,
    )


def get_micro_confusion_matrix_metrics(
    gold_output_intent: List[List[str]],
    pred_output_intent: List[List[str]],
    gold_output_slot: List[List[List[str]]],
    pred_output_slot: List[List[List[str]]],
    is_single_intent_detection: bool,
) -> Tuple[
    MicroConfusionMetrixMetricsModel, MicroConfusionMetrixMetricsProblemLevelModel
]:
    (
        confusion_matrix_intent_set,
        confusion_matrix_intent_set_list,
    ) = get_confision_matrix_from_answers(
        gold_answers=gold_output_intent,
        predicted_answers=pred_output_intent,
        is_single_intent_detection=is_single_intent_detection,
        mode=ConfusionMatrixMode.SET,
    )
    (
        confusion_matrix_intent_counter,
        confusion_matrix_intent_counter_list,
    ) = get_confision_matrix_from_answers(
        gold_answers=gold_output_intent,
        predicted_answers=pred_output_intent,
        is_single_intent_detection=is_single_intent_detection,
        mode=ConfusionMatrixMode.COUNTER,
    )
    (
        confusion_matrix_intent_list,
        confusion_matrix_intent_list_list,
    ) = get_confision_matrix_from_answers(
        gold_answers=gold_output_intent,
        predicted_answers=pred_output_intent,
        is_single_intent_detection=is_single_intent_detection,
        mode=ConfusionMatrixMode.LIST,
    )
    (
        confusion_matrix_slot_set,
        confusion_matrix_slot_set_list,
    ) = get_confision_matrix_from_answers_slot(
        gold_answers=gold_output_slot,
        predicted_answers=pred_output_slot,
        is_single_intent_detection=is_single_intent_detection,
        mode=ConfusionMatrixMode.SET,
    )
    return (
        MicroConfusionMetrixMetricsModel(
            intent_set_metrics=ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
                confusion_matrix_intent_set
            ),
            intent_counter_metrics=ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
                confusion_matrix_intent_counter
            ),
            intent_list_metrics=ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
                confusion_matrix_intent_list
            ),
            slot_set_metrics=ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
                confusion_matrix_slot_set
            ),
        ),
        MicroConfusionMetrixMetricsProblemLevelModel(
            intent_set_metrics_list=[
                ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
                    confusion_matrix_instance
                )
                for confusion_matrix_instance in confusion_matrix_intent_set_list
            ],
            intent_counter_metrics_list=[
                ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
                    confusion_matrix_instance
                )
                for confusion_matrix_instance in confusion_matrix_intent_counter_list
            ],
            intent_list_metrics_list=[
                ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
                    confusion_matrix_instance
                )
                for confusion_matrix_instance in confusion_matrix_intent_list_list
            ],
            slot_set_metrics_list=[
                ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
                    confusion_matrix_instance
                )
                for confusion_matrix_instance in confusion_matrix_slot_set_list
            ],
        ),
    )


def get_confusion_matrix_metrics_dict_by_output_length_from_dict(
    confusion_matrix_by_output_length_list_dict: Dict[int, List[ConfusionMatrixModel]],
) -> Dict[int, List[ConfusionMetrixMetricsModel]]:
    return {
        length: [
            ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
                confusion_matrix=matrix
            )
            for matrix in matrices
        ]
        for length, matrices in confusion_matrix_by_output_length_list_dict.items()
    }


def get_micro_confusion_matrix_metrics_by_output_length(
    gold_output_intent: List[List[str]],
    pred_output_intent: List[List[str]],
    gold_output_slot: List[List[List[str]]],
    pred_output_slot: List[List[List[str]]],
    is_single_intent_detection: bool,
) -> Tuple[
    MicroConfusionMetrixMetricsByOutputLengthModel,
    MicroConfusionMetrixMetricsByOutputLengthProblemLevelModel,
]:
    (
        confusion_matrix_intent_set_by_output_length_dict,
        confusion_matrix_intent_set_by_output_length_list_dict,
    ) = get_confision_matrix_from_answers_by_output_length(
        gold_answers=gold_output_intent,
        predicted_answers=pred_output_intent,
        is_single_intent_detection=is_single_intent_detection,
        mode=ConfusionMatrixMode.SET,
    )
    (
        confusion_matrix_intent_counter_by_output_length_dict,
        confusion_matrix_intent_counter_by_output_length_list_dict,
    ) = get_confision_matrix_from_answers_by_output_length(
        gold_answers=gold_output_intent,
        predicted_answers=pred_output_intent,
        is_single_intent_detection=is_single_intent_detection,
        mode=ConfusionMatrixMode.COUNTER,
    )
    (
        confusion_matrix_intent_list_by_output_length_dict,
        confusion_matrix_intent_list_by_output_length_list_dict,
    ) = get_confision_matrix_from_answers_by_output_length(
        gold_answers=gold_output_intent,
        predicted_answers=pred_output_intent,
        is_single_intent_detection=is_single_intent_detection,
        mode=ConfusionMatrixMode.LIST,
    )
    (
        confusion_matrix_slot_set_by_output_length_dict,
        confusion_matrix_slot_set_by_output_length_list_dict,
    ) = get_confision_matrix_from_answers_by_output_length_slot(
        gold_answers=gold_output_slot,
        predicted_answers=pred_output_slot,
        gold_output_intent=gold_output_intent,
        is_single_intent_detection=is_single_intent_detection,
        mode=ConfusionMatrixMode.SET,
    )
    X = MicroConfusionMetrixMetricsByOutputLengthModel(
        intent_set_metrics=ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro_by_output_length(
            confusion_matrix_intent_set_by_output_length_dict
        ),
        intent_counter_metrics=ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro_by_output_length(
            confusion_matrix_intent_counter_by_output_length_dict
        ),
        intent_list_metrics=ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro_by_output_length(
            confusion_matrix_intent_list_by_output_length_dict
        ),
        slot_set_metrics=ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro_by_output_length(
            confusion_matrix_slot_set_by_output_length_dict
        ),
    )
    Y = MicroConfusionMetrixMetricsByOutputLengthProblemLevelModel(
        intent_set_metrics_list=get_confusion_matrix_metrics_dict_by_output_length_from_dict(
            confusion_matrix_by_output_length_list_dict=confusion_matrix_intent_set_by_output_length_list_dict
        ),
        intent_counter_metrics_list=get_confusion_matrix_metrics_dict_by_output_length_from_dict(
            confusion_matrix_by_output_length_list_dict=confusion_matrix_intent_counter_by_output_length_list_dict
        ),
        intent_list_metrics_list=get_confusion_matrix_metrics_dict_by_output_length_from_dict(
            confusion_matrix_by_output_length_list_dict=confusion_matrix_intent_list_by_output_length_list_dict
        ),
        slot_set_metrics_list=get_confusion_matrix_metrics_dict_by_output_length_from_dict(
            confusion_matrix_by_output_length_list_dict=confusion_matrix_slot_set_by_output_length_list_dict
        ),
    )
    return (X, Y)


def parsing_only(
    predictions_input: List[EvaluationOutputResponseDataUnit],
    is_single_intent_detection: bool,
) -> List[EvaluationOutputResponseDataUnit]:
    parsed_outputs: List[EvaluationOutputResponseDataUnit] = []
    for datum in predictions_input:
        if is_single_intent_detection:
            (
                pred_func_calls,
                gold_func_calls,
                _,
                _,
                model_num_errors_parsing_pred_intent,
                _,
                _,
            ) = parse_output_from_language_models_rest(
                prediction=datum,
                model_name=datum.llm_model_id.split("/")[-1],
                is_single_intent_detection=is_single_intent_detection,
            )
        else:
            (
                pred_func_calls,
                gold_func_calls,
                _,
                _,
                model_num_errors_parsing_pred_intent,
                _,
                _,
            ) = parse_output_from_language_models(
                prediction=datum,
                model_name=datum.llm_model_id.split("/")[-1],
                is_single_intent_detection=is_single_intent_detection,
            )
        parsed_output = datum.model_copy(deep=True)
        parsed_output.predicted_function_calls = pred_func_calls
        parsed_output.gold_function_calls = gold_func_calls
        parsed_output.num_preciedtion_parsing_errors = (
            model_num_errors_parsing_pred_intent
        )
        parsed_outputs.append(parsed_output)

    return parsed_outputs


def calculate_scores(
    predictions_input: List[EvaluationOutputResponseDataUnit],
    is_single_intent_detection: bool = False,
) -> ScorerOuputModel:
    (
        model_name,
        _,
        spec_path,
        model_temperature,
        model_max_tokens,
        _,
    ) = predictions_input[0].get_dataset_basic_info()

    (
        gold_output_intent,
        pred_output_intent,
        gold_output_slot,
        pred_output_slot,
        gold_output_slot_problem,
        pred_output_slot_problem,
        num_errors_parsing_pred_intent,
        num_errors_parsing_gold_intent,
        num_errors_parsing_pred_slot,
        num_errors_parsing_gold_slot,
        all_num_times_full_score,
        num_pred_examples_w_parsing_errors,
        num_gold_examples_w_parsing_errors,
        error_messages,
        parsing_error_messages,
        sample_ids,
        predicted_function_calls,
        gold_function_calls,
        pred_dict_list,
        gold_dict_list,
    ) = get_item_metrics(
        predictions_input=predictions_input,
        model_name=model_name,
        is_single_intent_detection=is_single_intent_detection,
        spec_path=spec_path,
    )
    (
        confusion_metrix_matrics_micro_model,
        confusion_metrix_matrics_micro_problem_level_model,
    ) = get_micro_confusion_matrix_metrics(
        gold_output_intent=gold_output_intent,
        pred_output_intent=pred_output_intent,
        gold_output_slot=gold_output_slot_problem,
        pred_output_slot=pred_output_slot_problem,
        is_single_intent_detection=is_single_intent_detection,
    )
    (
        confusion_metrix_matrics_micro_model_by_output_length,
        confusion_metrix_matrics_micro_model_by_output_length_list,
    ) = get_micro_confusion_matrix_metrics_by_output_length(
        gold_output_intent=gold_output_intent,
        pred_output_intent=pred_output_intent,
        gold_output_slot=gold_output_slot_problem,
        pred_output_slot=pred_output_slot_problem,
        is_single_intent_detection=is_single_intent_detection,
    )

    num_samples = len(predictions_input)
    intent_pair_models = []
    pred_output_intent_new = []

    # {} in pred causes unhashable dict error (AGENTS)
    for gold, pred in zip(gold_output_intent, pred_output_intent):
        new_pred = clean_up_pred(pred)
        model = ContentPairModel(gold=gold, predicted=new_pred)
        intent_pair_models.append(model)
        pred_output_intent_new.append(new_pred)

    return ScorerOuputModel(
        confusion_metrix_matrics_micro_problem_level=confusion_metrix_matrics_micro_problem_level_model,
        confusion_metrix_matrics_micro_model_by_output_length_problem_level=confusion_metrix_matrics_micro_model_by_output_length_list,
        confusion_metrix_matrics_micro=confusion_metrix_matrics_micro_model,
        confusion_metrix_matrics_micro_model_by_output_length=confusion_metrix_matrics_micro_model_by_output_length,
        num_examples=num_samples,
        percentage_times_full_score=(all_num_times_full_score / num_samples),
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
        pred_output_intent=pred_output_intent_new,
        gold_output_slot=gold_output_slot,
        pred_output_slot=pred_output_slot,
        intent_pair_models=list(
            map(
                lambda content_pair: ContentPairModel(
                    gold=content_pair[0], predicted=content_pair[1]
                ),
                zip(gold_output_intent, pred_output_intent_new),
            )
        ),
        slot_pair_models=list(
            map(
                lambda content_pair: ContentPairModel(
                    gold=content_pair[0], predicted=content_pair[1]
                ),
                zip(gold_output_slot, pred_output_slot),
            )
        ),
        parsed_predictions=pred_dict_list,
        parsed_gold_answer=gold_dict_list,
    )


def handle_scoring_process_exception(
    output_root_path: Path,
    e: Exception,
    evaluator_output_file_path: Path,
    temperature_str: str,
    max_tokens_str: str,
    output_file_name: str,
) -> None:
    print(e)
    write_jsonl(
        file_path=Path(
            os.path.join(
                output_root_path,
                "error",
                temperature_str,
                max_tokens_str,
                output_file_name + "_" + get_uuid4_str() + ".json",
            )
        ),
        jsons=[CommonErrorModel(error=str(e), file=str(evaluator_output_file_path))],
        should_append=True,
    )


def parsing(
    evaluator_output_file_paths: List[Path],
    output_folder_path: Path,
    is_single_intent_detection: bool,
    ignore_file_path: Optional[Path] = None,
) -> Tuple[bool, int]:
    has_exception = False
    num_samples_ignored = 0
    sample_ignore_model: Optional[
        SampleIgonoreModel
    ] = get_sample_ignore_model_from_file(ignore_file_path=ignore_file_path)
    for evaluator_output_file_path in evaluator_output_file_paths:
        output_file_name = str(evaluator_output_file_path).split("/")[-1].split(".")[0]
        try:
            data, num_ignored = get_evaluation_output_response_data_units(
                evaluator_output_file_path=evaluator_output_file_path,
                sample_ignore_model=sample_ignore_model,
            )
            num_samples_ignored += num_ignored

            if data is None or len(data) == 0:
                raise Exception(
                    f"No evaluation data found at {evaluator_output_file_path}"
                )

            temperature_str, max_tokens_str, _, model_name, agent_str = data[
                0
            ].get_basic_strs()

            write_jsonl(
                file_path=Path(
                    os.path.join(
                        output_folder_path,
                        agent_str,
                        model_name,
                        temperature_str,
                        max_tokens_str,
                        (output_file_name + ".jsonl"),
                    )
                ),
                jsons=parsing_only(
                    predictions_input=data,
                    is_single_intent_detection=is_single_intent_detection,
                ),
                should_append=False,
            )

        except Exception as e:
            has_exception = True
            print(e)
            handle_scoring_process_exception(
                output_root_path=output_folder_path,
                e=e,
                evaluator_output_file_path=evaluator_output_file_path,
                temperature_str="default_temperature",
                max_tokens_str="default_max_tokens",
                output_file_name=output_file_name,
            )

    return has_exception, num_samples_ignored


def scoring(
    evaluator_output_file_paths: List[Path],
    output_folder_path: Path,
    is_single_intent_detection=False,
    ignore_file_path: Optional[Path] = None,
) -> Tuple[bool, int]:
    """
    returns the state of exception
    """

    has_exception = False
    num_samples_ignored = 0
    sample_ignore_model: Optional[
        SampleIgonoreModel
    ] = get_sample_ignore_model_from_file(ignore_file_path=ignore_file_path)

    for evaluator_output_file_path in evaluator_output_file_paths:
        temperature_str = "default_temperature"
        max_tokens_str = "default_max_tokens"
        model_name = "default_model"
        output_file_name = str(evaluator_output_file_path).split("/")[-1].split(".")[0]
        try:
            data, num_ignored = get_evaluation_output_response_data_units(
                evaluator_output_file_path=evaluator_output_file_path,
                sample_ignore_model=sample_ignore_model,
            )
            num_samples_ignored += num_ignored

            if data is None or len(data) == 0:
                raise Exception(
                    f"No evaluation data found at {evaluator_output_file_path}"
                )

            temperature_str, max_tokens_str, _, model_name, agent_str = data[
                0
            ].get_basic_strs()

            scorer_output_model = calculate_scores(
                data,
                is_single_intent_detection=is_single_intent_detection,
            )
            outfile = Path(
                os.path.join(
                    output_folder_path,
                    agent_str,
                    model_name.split("/")[-1],
                    temperature_str,
                    max_tokens_str,
                    (output_file_name + ".json"),
                )
            )
            write_json(file_path=outfile, base_model=scorer_output_model)
        except Exception as e:
            has_exception = True
            handle_scoring_process_exception(
                output_root_path=output_folder_path,
                e=e,
                evaluator_output_file_path=evaluator_output_file_path,
                temperature_str=temperature_str,
                max_tokens_str=max_tokens_str,
                output_file_name=output_file_name,
            )
    return has_exception, num_samples_ignored
