import os
from pathlib import Path
from typing import Any, List, Optional, Tuple, cast

from api_integrated_llm.data_models.scorer_models import WinRateResultModel
from api_integrated_llm.data_models.source_models import (
    EvaluationOutputResponseDataUnit,
    QuerySourceModel,
)
from api_integrated_llm.helpers.database_helper.metrics_helper.win_rate_helper import (
    evaluate_win_rate,
    get_payloads_winrate,
    setup,
)
from api_integrated_llm.helpers.file_helper import (
    get_base_model_from_json,
    get_files_in_folder,
)


def get_source_file_path(
    response_units: List[EvaluationOutputResponseDataUnit],
    source_file_search_path: Path,
) -> str:
    source_file_path = response_units[0].source_file_path
    candidtate_source_file_paths = get_files_in_folder(
        folder_path=Path(os.path.join(source_file_search_path)),
        file_extension="json",
    )
    source_file_name = source_file_path.split("/")[-1]

    for candidate_source_file_path in candidtate_source_file_paths:
        file_path_str = str(candidate_source_file_path)
        if source_file_name in file_path_str:
            source_file_path = file_path_str
            break
    return source_file_path


def get_dataset_name(source_file_path: str) -> str:
    dataset_name = ""
    source_obj = cast(
        QuerySourceModel,
        get_base_model_from_json(
            file_path=source_file_path, base_model=QuerySourceModel
        ),
    )

    if source_obj.dataset:
        dataset_name = source_obj.dataset[:]
    else:
        raise Exception(f'"dataset" is not defined at {source_file_path}')

    return dataset_name


def get_win_rate(
    predictions_input: List[EvaluationOutputResponseDataUnit],
    predicted_function_calls: List[List[Any]],
    gold_function_calls: List[List[Any]],
    sample_ids: List[Any],
    db_path: Path,  # database path
    source_file_search_path: Path,
) -> Tuple[Optional[float], int, List[str], List[int], WinRateResultModel]:
    """
    returns winrate, the number of sequences evaluated, and error messages
    """

    win_rate: Optional[float] = None
    error_messages: List[str] = []
    num_failed_function_execution_list: List[int] = []
    win_rate_result_model = WinRateResultModel()

    if len(predicted_function_calls) != len(sample_ids):
        error_messages.append(
            "The number of sample_IDs do not match the number of"
            + " predicted function calls at get_win_rate()"
        )
        return win_rate, 0, error_messages

    try:
        source_file_path = get_source_file_path(
            response_units=predictions_input,
            source_file_search_path=source_file_search_path,
        )

        dataset_name = get_dataset_name(source_file_path=source_file_path)
        builder, cache_folder_path = setup(source_file_path, db_path)
        predicted_function_calls_tuple = [
            (sample_id, function_calls)
            for sample_id, function_calls in zip(sample_ids, predicted_function_calls)
        ]
        payloads, error_messages_instance = get_payloads_winrate(
            source_model=get_base_model_from_json(
                file_path=Path(source_file_path),
                base_model=QuerySourceModel,
            ),
            cache_folder_path=Path(cache_folder_path),
            dataset_name=dataset_name,
            predicted_function_calls_tuple=predicted_function_calls_tuple,
        )
        error_messages.extend(error_messages_instance)

        if len(payloads) > 0:
            (
                win_rate,
                error_messages,
                num_failed_function_execution,
                win_rate_result_model,
            ) = evaluate_win_rate(
                payloads,
                builder,
                pred_function_calls_list=predicted_function_calls,
                gold_function_calls_list=gold_function_calls,
            )
            error_messages.extend(error_messages)
            num_failed_function_execution_list.append(num_failed_function_execution)
    except Exception as e:
        error_messages.append(str(e))

    return (
        win_rate,
        len(payloads),
        error_messages,
        num_failed_function_execution_list,
        win_rate_result_model,
    )
