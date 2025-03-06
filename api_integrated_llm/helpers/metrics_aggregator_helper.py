from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, cast

from api_integrated_llm.data_models.metrics_models import (
    DefaultMetricsAggregationConfiguration,
)
from api_integrated_llm.data_models.scorer_models import (
    ConfusionMatrixModel,
    ConfusionMetrixMetricsModel,
    MetaMetricsAggregationModel,
    MetricsAggregationModel,
    MicroConfusionMetrixMetricsByOutputLengthContainerModel,
    MicroConfusionMetrixMetricsModel,
    ScorerOuputModel,
)
from api_integrated_llm.helpers.file_helper import (
    get_base_models_from_folder_tuple,
    get_dict_from_json,
)


def get_llm_model_names(metrics_objs: List[Tuple[Path, ScorerOuputModel]]) -> List[str]:
    llm_model_names: Set[str] = set()
    for _, obj in metrics_objs:
        if len(obj.evaluation_source) > 0:
            llm_model_names.add(obj.evaluation_source[0].llm_model_id.lower())

    return list(llm_model_names)


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
    metrics_configuration_obj["file_name"] = list(
        set([str(tmp_path).split("/")[-1] for tmp_path, _ in metrics_objs])
    )
    metrics_configuration_obj["llm"] = get_llm_model_names(metrics_objs=metrics_objs)

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


def get_output_length_metrics_categories_dict(
    path_model_list: List[Tuple[Path, ScorerOuputModel]],
) -> MicroConfusionMetrixMetricsByOutputLengthContainerModel:
    intent_set_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = {}
    intent_counter_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = {}
    intent_list_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = {}
    slot_set_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = {}

    for _, score_output_model in path_model_list:
        meta_model = (
            score_output_model.confusion_metrix_matrics_micro_model_by_output_length
        )

        if meta_model is not None:
            for frequency, metrics_model in meta_model.intent_set_metrics.items():
                frequency_str = str(frequency)
                if frequency_str not in intent_set_metrics:
                    intent_set_metrics[frequency_str] = [metrics_model]
                else:
                    intent_set_metrics[frequency_str].append(metrics_model)

            for frequency, metrics_model in meta_model.intent_counter_metrics.items():
                frequency_str = str(frequency)
                if frequency_str not in intent_counter_metrics:
                    intent_counter_metrics[frequency_str] = [metrics_model]
                else:
                    intent_counter_metrics[frequency_str].append(metrics_model)

            for frequency, metrics_model in meta_model.intent_list_metrics.items():
                frequency_str = str(frequency)
                if frequency_str not in intent_list_metrics:
                    intent_list_metrics[frequency_str] = [metrics_model]
                else:
                    intent_list_metrics[frequency_str].append(metrics_model)

            for frequency, metrics_model in meta_model.slot_set_metrics.items():
                frequency_str = str(frequency)
                if frequency_str not in slot_set_metrics:
                    slot_set_metrics[frequency_str] = [metrics_model]
                else:
                    slot_set_metrics[frequency_str].append(metrics_model)

    return MicroConfusionMetrixMetricsByOutputLengthContainerModel(
        intent_set_metrics=intent_set_metrics,
        intent_counter_metrics=intent_counter_metrics,
        intent_list_metrics=intent_list_metrics,
        slot_set_metrics=slot_set_metrics,
    )


def get_agent_metrics_categories_dict(
    path_model_list: List[Tuple[Path, ScorerOuputModel]],
) -> Dict[str, List[MicroConfusionMetrixMetricsModel]]:
    agent = "agent"
    llm = "llm"
    metrics_categories_dict: Dict[str, List[MicroConfusionMetrixMetricsModel]] = {
        agent: [],
        llm: [],
    }

    for _, score_output_model in path_model_list:
        if (
            len(score_output_model.evaluation_source) > 0
            and score_output_model.evaluation_source[0].is_agent
        ):
            metrics_categories_dict[agent].append(
                score_output_model.confusion_metrix_matrics_micro.model_copy(deep=True)
            )
        else:
            metrics_categories_dict[llm].append(
                score_output_model.confusion_metrix_matrics_micro.model_copy(deep=True)
            )

    return metrics_categories_dict


def get_category_metrics_categories_dict(
    path_model_list: List[Tuple[Path, ScorerOuputModel]], categories: List[str]
) -> Dict[str, List[MicroConfusionMetrixMetricsModel]]:
    lowered_categories = [category.lower() for category in categories]
    metrics_categories_dict: Dict[str, List[MicroConfusionMetrixMetricsModel]] = {
        category: [] for category in lowered_categories
    }

    for path, score_output_model in path_model_list:
        path_str = str(path).lower()
        for category in lowered_categories:
            if category in path_str:
                metrics_categories_dict[category].append(
                    score_output_model.confusion_metrix_matrics_micro.model_copy(
                        deep=True
                    )
                )

    return metrics_categories_dict


def get_meta_metrics_dict(
    metrics_categories_dict: Dict[str, List[MicroConfusionMetrixMetricsModel]],
) -> Tuple[
    Dict[str, List[ConfusionMetrixMetricsModel]],
    Dict[str, List[ConfusionMetrixMetricsModel]],
    Dict[str, List[ConfusionMetrixMetricsModel]],
    Dict[str, List[ConfusionMetrixMetricsModel]],
]:
    categories = list(metrics_categories_dict.keys())
    intent_set_dict: Dict[str, List[ConfusionMetrixMetricsModel]] = {
        category: [] for category in categories
    }
    intent_counter_dict: Dict[str, List[ConfusionMetrixMetricsModel]] = {
        category: [] for category in categories
    }
    intent_list_dict: Dict[str, List[ConfusionMetrixMetricsModel]] = {
        category: [] for category in categories
    }
    slot_set_dict: Dict[str, List[ConfusionMetrixMetricsModel]] = {
        category: [] for category in categories
    }

    for category, model_list in metrics_categories_dict.items():
        for model in model_list:
            intent_set_dict[category].append(
                model.intent_set_metrics.model_copy(deep=True)
            )
            intent_counter_dict[category].append(
                model.intent_counter_metrics.model_copy(deep=True)
            )
            intent_list_dict[category].append(
                model.intent_list_metrics.model_copy(deep=True)
            )
            slot_set_dict[category].append(model.slot_set_metrics.model_copy(deep=True))

    return intent_set_dict, intent_counter_dict, intent_list_dict, slot_set_dict


def get_meta_metrics(
    intent_set_dict: Dict[str, List[ConfusionMetrixMetricsModel]],
    intent_counter_dict: Dict[str, List[ConfusionMetrixMetricsModel]],
    intent_list_dict: Dict[str, List[ConfusionMetrixMetricsModel]],
    slot_set_dict: Dict[str, List[ConfusionMetrixMetricsModel]],
    categories: Optional[List[str]] = None,
) -> MetaMetricsAggregationModel:
    output_categories = deepcopy(categories) if categories is not None else []
    return MetaMetricsAggregationModel(
        intent_set_metrics=MetricsAggregationModel(
            micro=get_micro_metrics_aggregation_dict(
                categories_dict=intent_set_dict,
            ),
            macro=get_macro_metrics_aggregation_dict(
                categories_dict=intent_set_dict,
            ),
            categories=output_categories,
            raw_data=intent_set_dict,
        ),
        intent_counter_metrics=MetricsAggregationModel(
            micro=get_micro_metrics_aggregation_dict(
                categories_dict=intent_counter_dict,
            ),
            macro=get_macro_metrics_aggregation_dict(
                categories_dict=intent_counter_dict,
            ),
            categories=output_categories,
            raw_data=intent_counter_dict,
        ),
        intent_list_metrics=MetricsAggregationModel(
            micro=get_micro_metrics_aggregation_dict(
                categories_dict=intent_list_dict,
            ),
            macro=get_macro_metrics_aggregation_dict(
                categories_dict=intent_list_dict,
            ),
            categories=output_categories,
            raw_data=intent_list_dict,
        ),
        slot_set_metrics=MetricsAggregationModel(
            micro=get_micro_metrics_aggregation_dict(
                categories_dict=slot_set_dict,
            ),
            macro=get_macro_metrics_aggregation_dict(
                categories_dict=slot_set_dict,
            ),
            categories=output_categories,
            raw_data=slot_set_dict,
        ),
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
) -> Optional[MetricsAggregationModel]:
    categories_dict, has_error = get_categories_dict(
        metircs_obj_list=metircs_obj_list,
        categories=categories,
    )

    if has_error:
        return None

    return MetricsAggregationModel(
        micro=get_micro_metrics_aggregation_dict(
            categories_dict=categories_dict,
        ),
        macro=get_macro_metrics_aggregation_dict(
            categories_dict=categories_dict,
        ),
        categories=deepcopy(categories),
        raw_data=categories_dict,
    )


def get_output_length_meta_metrics_aggregation_model(
    path_model_list: List[Tuple[Path, ScorerOuputModel]],
) -> MetaMetricsAggregationModel:
    container_model = get_output_length_metrics_categories_dict(
        path_model_list=path_model_list,
    )

    return get_meta_metrics(
        intent_set_dict=container_model.intent_set_metrics,
        intent_counter_dict=container_model.intent_counter_metrics,
        intent_list_dict=container_model.intent_list_metrics,
        slot_set_dict=container_model.slot_set_metrics,
    )


def get_agent_meta_metrics_aggregation_model(
    path_model_list: List[Tuple[Path, ScorerOuputModel]],
) -> MetaMetricsAggregationModel:
    metrics_categories_dict = get_agent_metrics_categories_dict(
        path_model_list=path_model_list
    )
    categories = list(metrics_categories_dict.keys())
    (
        intent_set_dict,
        intent_counter_dict,
        intent_list_dict,
        slot_set_dict,
    ) = get_meta_metrics_dict(metrics_categories_dict=metrics_categories_dict)

    return get_meta_metrics(
        intent_set_dict=intent_set_dict,
        intent_counter_dict=intent_counter_dict,
        intent_list_dict=intent_list_dict,
        slot_set_dict=slot_set_dict,
        categories=categories,
    )


def get_category_meta_metrics_aggregation_model(
    path_model_list: List[Tuple[Path, ScorerOuputModel]], categories: List[str]
) -> MetaMetricsAggregationModel:
    metrics_categories_dict = get_category_metrics_categories_dict(
        path_model_list=path_model_list, categories=categories
    )
    (
        intent_set_dict,
        intent_counter_dict,
        intent_list_dict,
        slot_set_dict,
    ) = get_meta_metrics_dict(metrics_categories_dict=metrics_categories_dict)

    return get_meta_metrics(
        intent_set_dict=intent_set_dict,
        intent_counter_dict=intent_counter_dict,
        intent_list_dict=intent_list_dict,
        slot_set_dict=slot_set_dict,
        categories=categories,
    )
