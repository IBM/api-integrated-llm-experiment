from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

from api_integrated_llm.data_models.metrics_models import (
    DefaultMetricsAggregationConfiguration,
)
from api_integrated_llm.data_models.scorer_models import (
    ConfusionMatrixModel,
    ConfusionMetrixMetricsModel,
    MetricsAggregationModel,
    ScorerOuputModel,
)
from api_integrated_llm.helpers.file_helper import (
    get_base_models_from_folder_tuple,
    get_dict_from_json,
)


def get_metrics_aggregator_inputs(
    scoring_output_folder_path: Optional[Path],
    metrics_aggregation_configuration_file_path: Optional[Path],
) -> Tuple[List[Tuple[Path, ScorerOuputModel]], Dict[str, List[str]], bool]:
    has_error = False
    if scoring_output_folder_path is None:
        has_error = True
        print("No scoring Output Folder Path is provided")
        return [], {}, has_error

    metrics_objs = cast(
        List[Tuple[Path, ScorerOuputModel]],
        get_base_models_from_folder_tuple(
            folder_path=scoring_output_folder_path,
            file_extension="json",
            base_model=ScorerOuputModel,
        ),
    )

    metrics_configuration_obj = DefaultMetricsAggregationConfiguration().get_dict()

    if metrics_aggregation_configuration_file_path is not None:
        try:
            metrics_configuration_obj = get_dict_from_json(
                file_path=metrics_aggregation_configuration_file_path
            )

            for v in metrics_configuration_obj.values():
                if not isinstance(v, list):
                    raise Exception("Invalid metrics aggregation category type")

        except Exception as e:
            has_error = True
            print(f"Metrics Aggregation Configuration File Parsing failed: {str(e)}")
    return (
        metrics_objs,
        cast(Dict[str, List[str]], metrics_configuration_obj),
        has_error,
    )


def get_categories_dict(
    metircs_obj_list: List[Tuple[Path, ConfusionMetrixMetricsModel]],
    categories: List[str],
) -> Tuple[Dict[str, List[ConfusionMetrixMetricsModel]], bool]:
    has_error = False

    if len(categories) == 0:
        has_error = True
        return dict(), has_error

    categories_dict: Dict[str, List[ConfusionMetrixMetricsModel]] = {
        category: [] for category in categories
    }

    for metrics_file_path, metrics_obj in metircs_obj_list:
        metrics_file_path_str = str(metrics_file_path)

        for category in categories:
            if category in metrics_file_path_str:
                categories_dict[category].append(metrics_obj.model_copy(deep=True))
                break

    return categories_dict, has_error


def get_micro_metrics_aggregation_dict(
    categories_dict: Dict[str, List[ConfusionMetrixMetricsModel]],
) -> Dict[str, ConfusionMetrixMetricsModel]:
    metrics_aggregation_dict: Dict[str, ConfusionMetrixMetricsModel] = {
        category: ConfusionMetrixMetricsModel() for category in categories_dict.keys()
    }

    for category, metrics_model_list in categories_dict.items():
        confusion_metrics_model_acc = (
            ConfusionMatrixModel(mode=metrics_model_list[0].confusion_matrix.mode)
            if (
                len(metrics_model_list) > 0
                and metrics_model_list[0].confusion_matrix is not None
            )
            else ConfusionMatrixModel()
        )

        for metrics_model in metrics_model_list:
            if metrics_model.confusion_matrix is not None:
                confusion_metrics_model_acc.add(
                    confusion_matrix=metrics_model.confusion_matrix
                )
        metrics_model = ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
            confusion_matrix=confusion_metrics_model_acc
        )
        metrics_model.set_f1()
        metrics_aggregation_dict[category] = metrics_model

    return metrics_aggregation_dict


def get_macro_metrics_aggregation_dict(
    categories_dict: Dict[str, List[ConfusionMetrixMetricsModel]],
) -> Dict[str, ConfusionMetrixMetricsModel]:
    metrics_aggregation_dict: Dict[str, ConfusionMetrixMetricsModel] = {
        category: ConfusionMetrixMetricsModel() for category in categories_dict.keys()
    }

    for category, metrics_model_list in categories_dict.items():
        metrics_model_acc = ConfusionMetrixMetricsModel(
            accuracy=0.0, precision=0.0, recall=0.0, f1=0.0
        )
        num_samples = len(metrics_model_list)
        for metrics_model in metrics_model_list:
            metrics_model_acc.add(metrics_model=metrics_model, num_samples=num_samples)

        metrics_model_acc.set_f1()
        metrics_aggregation_dict[category] = metrics_model_acc

    return metrics_aggregation_dict


def get_aggregated_metrics(
    metircs_obj_list: List[Tuple[Path, ConfusionMetrixMetricsModel]],
    categories: List[str],
) -> Tuple[MetricsAggregationModel, bool]:
    categories_dict, has_error = get_categories_dict(
        metircs_obj_list=metircs_obj_list,
        categories=categories,
    )

    if has_error:
        return MetricsAggregationModel(), has_error

    return (
        MetricsAggregationModel(
            micro=get_micro_metrics_aggregation_dict(
                categories_dict=categories_dict,
            ),
            macro=get_macro_metrics_aggregation_dict(
                categories_dict=categories_dict,
            ),
            categories=deepcopy(categories),
            raw_data=categories_dict,
        ),
        has_error,
    )
