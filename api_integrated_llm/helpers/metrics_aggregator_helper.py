from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, cast

from api_integrated_llm.data_models.metrics_models import (
    DefaultMetricsAggregationConfiguration,
)
from api_integrated_llm.data_models.scorer_models import (
    BasicRateDictMetaModel,
    BasicRateDictModel,
    BasicRateModel,
    BasicRateUnitModel,
    ConfusionMatrixModel,
    ConfusionMetrixMetricsModel,
    MetaMetricsAggregationModel,
    MetricsAggregationModel,
    MicroConfusionMetrixMetricsByOutputLengthContainerModel,
    MicroConfusionMetrixMetricsModel,
    MicroConfusionMetrixMetricsProblemLevelModel,
    ScorerOuputModel,
    WinRateResultModel,
    WinRateResultUnitModel,
)
from api_integrated_llm.helpers.file_helper import (
    get_base_models_from_folder_tuple,
    get_dict_from_json,
)


def get_llm_model_names(metrics_objs: List[Tuple[Path, ScorerOuputModel]]) -> List[str]:
    llm_model_names: Set[str] = set()
    for _, obj in metrics_objs:
        if len(obj.evaluation_source) > 0:
            name = obj.evaluation_source[0].llm_model_id.split("/")[-1].lower()
            llm_model_names.add(name)

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
) -> Tuple[
    MicroConfusionMetrixMetricsByOutputLengthContainerModel,
    MicroConfusionMetrixMetricsByOutputLengthContainerModel,
    Dict[str, List[WinRateResultModel]],
]:
    return (
        get_output_length_metrics_categories_dict_micro_metrics(
            path_model_list=path_model_list,
        ),
        get_output_length_metrics_categories_dict_problem_level(
            path_model_list=path_model_list,
        ),
        get_output_length_metrics_categories_dict_win_rate(
            path_model_list=path_model_list,
        ),
    )


def get_output_length_metrics_categories_dict_micro_metrics(
    path_model_list: List[Tuple[Path, ScorerOuputModel]],
) -> MicroConfusionMetrixMetricsByOutputLengthContainerModel:
    intent_set_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = {}
    intent_counter_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = {}
    intent_list_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = {}
    slot_set_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = {}

    for _, score_output_model in path_model_list:
        # confusion matrix
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


def get_output_length_metrics_categories_dict_win_rate(
    path_model_list: List[Tuple[Path, ScorerOuputModel]],
) -> Dict[str, List[WinRateResultModel]]:
    win_rate_result_dict: Dict[str, List[WinRateResultModel]] = dict()

    for _, score_output_model in path_model_list:
        # win rate
        if (
            score_output_model.win_rate_result_model is not None
            and len(score_output_model.win_rate_result_model.win_rate_result) > 0
        ):
            basic_rate_unit_dict_model: Dict[str, List[WinRateResultUnitModel]] = dict()
            for (
                win_rate_result_unit_model
            ) in score_output_model.win_rate_result_model.win_rate_result:
                gold_sequence_length_str = str(
                    len(win_rate_result_unit_model.gold_function_calls)
                )

                if gold_sequence_length_str not in basic_rate_unit_dict_model:
                    basic_rate_unit_dict_model[gold_sequence_length_str] = list()

                basic_rate_unit_dict_model[gold_sequence_length_str].append(
                    win_rate_result_unit_model.model_copy(deep=True)
                )

            for (
                frequency_str,
                win_rate_result_unit_model_list,
            ) in basic_rate_unit_dict_model.items():
                if frequency_str not in win_rate_result_dict:
                    win_rate_result_dict[frequency_str] = list()
                win_rate_result_dict[frequency_str].append(
                    WinRateResultModel(win_rate_result=win_rate_result_unit_model_list)
                )

    return win_rate_result_dict


def get_output_length_metrics_categories_dict_problem_level(
    path_model_list: List[Tuple[Path, ScorerOuputModel]],
) -> MicroConfusionMetrixMetricsByOutputLengthContainerModel:
    intent_set_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = {}
    intent_counter_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = {}
    intent_list_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = {}
    slot_set_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = {}

    for _, score_output_model in path_model_list:
        # confusion matrix
        meta_model = (
            score_output_model.confusion_metrix_matrics_micro_model_by_output_length_problem_level
        )

        if meta_model is not None:
            for frequency, metrics_models in meta_model.intent_set_metrics_list.items():
                frequency_str = str(frequency)
                if frequency_str not in intent_set_metrics:
                    intent_set_metrics[frequency_str] = [
                        metrics_model.model_copy(deep=True)
                        for metrics_model in metrics_models
                    ]
                else:
                    intent_set_metrics[frequency_str].extend(metrics_models)

            for (
                frequency,
                metrics_model,
            ) in meta_model.intent_counter_metrics_list.items():
                frequency_str = str(frequency)
                if frequency_str not in intent_counter_metrics:
                    intent_counter_metrics[frequency_str] = [
                        metrics_model.model_copy(deep=True)
                        for metrics_model in metrics_models
                    ]
                else:
                    intent_counter_metrics[frequency_str].extend(metrics_model)

            for frequency, metrics_model in meta_model.intent_list_metrics_list.items():
                frequency_str = str(frequency)
                if frequency_str not in intent_list_metrics:
                    intent_list_metrics[frequency_str] = [
                        metrics_model.model_copy(deep=True)
                        for metrics_model in metrics_models
                    ]
                else:
                    intent_list_metrics[frequency_str].extend(metrics_model)

            for frequency, metrics_model in meta_model.slot_set_metrics_list.items():
                frequency_str = str(frequency)
                if frequency_str not in slot_set_metrics:
                    slot_set_metrics[frequency_str] = [
                        metrics_model.model_copy(deep=True)
                        for metrics_model in metrics_models
                    ]
                else:
                    slot_set_metrics[frequency_str].extend(metrics_model)

    return MicroConfusionMetrixMetricsByOutputLengthContainerModel(
        intent_set_metrics=intent_set_metrics,
        intent_counter_metrics=intent_counter_metrics,
        intent_list_metrics=intent_list_metrics,
        slot_set_metrics=slot_set_metrics,
    )


def get_agent_metrics_categories_dict(
    path_model_list: List[Tuple[Path, ScorerOuputModel]],
) -> Tuple[
    Dict[str, List[MicroConfusionMetrixMetricsModel]],
    Dict[str, MicroConfusionMetrixMetricsProblemLevelModel],
    Dict[str, List[WinRateResultModel]],
]:
    return (
        get_agent_metrics_categories_dict_general(
            path_model_list=path_model_list,
        ),
        get_agent_metrics_categories_dict_problem_level(
            path_model_list=path_model_list,
        ),
        get_agent_metrics_categories_dict_win_rate(
            path_model_list=path_model_list,
        ),
    )


def get_agent_metrics_categories_dict_general(
    path_model_list: List[Tuple[Path, ScorerOuputModel]],
) -> Dict[str, List[MicroConfusionMetrixMetricsModel]]:
    agent = "agent"
    llm = "llm"
    metrics_categories_dict: Dict[str, List[MicroConfusionMetrixMetricsModel]] = {
        agent: [],
        llm: [],
    }

    for _, score_output_model in path_model_list:
        category = (
            agent
            if (
                len(score_output_model.evaluation_source) > 0
                and score_output_model.evaluation_source[0].is_agent
            )
            else llm
        )

        metrics_categories_dict[category].append(
            score_output_model.confusion_metrix_matrics_micro.model_copy(deep=True)
        )

    return metrics_categories_dict


def get_agent_metrics_categories_dict_problem_level(
    path_model_list: List[Tuple[Path, ScorerOuputModel]],
) -> Dict[str, MicroConfusionMetrixMetricsProblemLevelModel]:
    agent = "agent"
    llm = "llm"
    metrics_categories_dict: Dict[str, MicroConfusionMetrixMetricsProblemLevelModel] = {
        agent: MicroConfusionMetrixMetricsProblemLevelModel(),
        llm: MicroConfusionMetrixMetricsProblemLevelModel(),
    }

    for _, score_output_model in path_model_list:
        if score_output_model.confusion_metrix_matrics_micro_problem_level is not None:
            category = (
                agent
                if (
                    len(score_output_model.evaluation_source) > 0
                    and score_output_model.evaluation_source[0].is_agent
                )
                else llm
            )

            meta_model = score_output_model.confusion_metrix_matrics_micro_problem_level
            meta_model.intent_set_metrics_list
            metrics_categories_dict[category].intent_set_metrics_list.extend(
                meta_model.intent_set_metrics_list
            )
            metrics_categories_dict[category].intent_counter_metrics_list.extend(
                meta_model.intent_counter_metrics_list
            )
            metrics_categories_dict[category].intent_list_metrics_list.extend(
                meta_model.intent_list_metrics_list
            )
            metrics_categories_dict[category].slot_set_metrics_list.extend(
                meta_model.slot_set_metrics_list
            )

    return metrics_categories_dict


def get_agent_metrics_categories_dict_win_rate(
    path_model_list: List[Tuple[Path, ScorerOuputModel]],
) -> Dict[str, List[WinRateResultModel]]:
    agent = "agent"
    llm = "llm"
    win_rate_categories_dict: Dict[str, List[WinRateResultModel]] = {
        agent: [],
        llm: [],
    }

    for _, score_output_model in path_model_list:
        category = (
            agent
            if (
                len(score_output_model.evaluation_source) > 0
                and score_output_model.evaluation_source[0].is_agent
            )
            else llm
        )

        if score_output_model.win_rate_result_model is not None:
            win_rate_categories_dict[category].append(
                score_output_model.win_rate_result_model.model_copy(deep=True)
            )

    return win_rate_categories_dict


def get_category_metrics_categories_dict(
    path_model_list: List[Tuple[Path, ScorerOuputModel]], categories: List[str]
) -> Tuple[
    Dict[str, List[MicroConfusionMetrixMetricsModel]],
    Dict[str, List[MicroConfusionMetrixMetricsProblemLevelModel]],
    Dict[str, List[WinRateResultModel]],
]:
    lowered_categories = [category.lower() for category in categories]
    metrics_categories_dict: Dict[str, List[MicroConfusionMetrixMetricsModel]] = {
        category: [] for category in lowered_categories
    }
    metrics_categories_dict_problem_level: Dict[
        str, List[MicroConfusionMetrixMetricsProblemLevelModel]
    ] = {category: [] for category in lowered_categories}
    win_rate_categories_dict: Dict[str, List[WinRateResultModel]] = {
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
                if (
                    score_output_model.confusion_metrix_matrics_micro_problem_level
                    is not None
                ):
                    metrics_categories_dict_problem_level[category].append(
                        score_output_model.confusion_metrix_matrics_micro_problem_level
                    )
                if (
                    score_output_model.win_rate_result_model is not None
                    and len(score_output_model.win_rate_result_model.win_rate_result)
                    > 0
                ):
                    win_rate_categories_dict[category].append(
                        WinRateResultModel(
                            win_rate_result=deepcopy(
                                score_output_model.win_rate_result_model.win_rate_result
                            )
                        )
                    )

    return (
        metrics_categories_dict,
        metrics_categories_dict_problem_level,
        win_rate_categories_dict,
    )


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


def get_win_rate_aggregation_dict(
    categories_dict: Dict[str, List[WinRateResultModel]],
) -> BasicRateDictMetaModel:
    micro_aggregation_model = BasicRateDictModel()
    macro_aggregation_model = BasicRateDictModel()
    for category, win_rate_result_models in categories_dict.items():
        if len(win_rate_result_models) == 0:
            continue

        if category not in micro_aggregation_model.rate_dictionary:
            micro_aggregation_model.rate_dictionary[category] = BasicRateModel()
            micro_aggregation_model.raw_data[category] = []
            macro_aggregation_model.rate_dictionary[category] = BasicRateModel()
            macro_aggregation_model.raw_data[category] = []

        num_win_rate_result_models = len(win_rate_result_models)
        for win_rate_result_model in win_rate_result_models:
            if len(win_rate_result_model.win_rate_result) == 0:
                continue

            basic_rate_unit_model = BasicRateUnitModel()

            for win_rate_result_unit_model in win_rate_result_model.win_rate_result:
                micro_aggregation_model.raw_data[category].append(
                    win_rate_result_unit_model.model_copy(deep=True)
                )
                macro_aggregation_model.raw_data[category].append(
                    win_rate_result_unit_model.model_copy(deep=True)
                )
                basic_rate_unit_model.add(
                    unit_model=win_rate_result_unit_model.get_basic_rate_unit_model()
                )

            basic_rate_model = BasicRateModel.get_basic_rate_model(
                unit_model=basic_rate_unit_model
            )
            micro_aggregation_model.rate_dictionary[category].add_micro(
                rate_model=basic_rate_model
            )
            macro_aggregation_model.rate_dictionary[category].add_macro(
                rate_model=basic_rate_model,
                num_samples=num_win_rate_result_models,
            )
    return BasicRateDictMetaModel(
        micro_rate=micro_aggregation_model, macro_rate=macro_aggregation_model
    )


def get_meta_metrics(
    intent_set_dict: Dict[str, List[ConfusionMetrixMetricsModel]],
    intent_counter_dict: Dict[str, List[ConfusionMetrixMetricsModel]],
    intent_list_dict: Dict[str, List[ConfusionMetrixMetricsModel]],
    slot_set_dict: Dict[str, List[ConfusionMetrixMetricsModel]],
    win_rate_categories_dict: Optional[Dict[str, List[WinRateResultModel]]] = None,
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
        win_rate_metrics=(
            get_win_rate_aggregation_dict(categories_dict=win_rate_categories_dict)
            if win_rate_categories_dict is not None
            else None
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
        metrics_aggregation_dict[category] = (
            metrics_model_acc
            if len(metrics_model_list) > 0
            else ConfusionMetrixMetricsModel()
        )

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
) -> Tuple[MetaMetricsAggregationModel, MetaMetricsAggregationModel]:
    (
        container_model,
        container_model_problem_level,
        win_rate_result_dict,
    ) = get_output_length_metrics_categories_dict(
        path_model_list=path_model_list,
    )

    return (
        get_meta_metrics(
            intent_set_dict=container_model.intent_set_metrics,
            intent_counter_dict=container_model.intent_counter_metrics,
            intent_list_dict=container_model.intent_list_metrics,
            slot_set_dict=container_model.slot_set_metrics,
            win_rate_categories_dict=win_rate_result_dict,
        ),
        get_meta_metrics(
            intent_set_dict=container_model_problem_level.intent_set_metrics,
            intent_counter_dict=container_model_problem_level.intent_counter_metrics,
            intent_list_dict=container_model_problem_level.intent_list_metrics,
            slot_set_dict=container_model_problem_level.slot_set_metrics,
        ),
    )


def get_agent_meta_metrics_aggregation_model(
    path_model_list: List[Tuple[Path, ScorerOuputModel]],
) -> Tuple[MetaMetricsAggregationModel, MetaMetricsAggregationModel]:
    (
        metrics_categories_dict,
        metrics_categories_dict_problem_level,
        win_rate_categories_dict,
    ) = get_agent_metrics_categories_dict(path_model_list=path_model_list)
    categories = list(metrics_categories_dict.keys())
    (
        intent_set_dict,
        intent_counter_dict,
        intent_list_dict,
        slot_set_dict,
    ) = get_meta_metrics_dict(metrics_categories_dict=metrics_categories_dict)

    return (
        get_meta_metrics(
            intent_set_dict=intent_set_dict,
            intent_counter_dict=intent_counter_dict,
            intent_list_dict=intent_list_dict,
            slot_set_dict=slot_set_dict,
            win_rate_categories_dict=win_rate_categories_dict,
            categories=categories,
        ),
        get_meta_metrics(
            intent_set_dict={
                category: content.intent_set_metrics_list
                for category, content in metrics_categories_dict_problem_level.items()
            },
            intent_counter_dict={
                category: content.intent_counter_metrics_list
                for category, content in metrics_categories_dict_problem_level.items()
            },
            intent_list_dict={
                category: content.intent_list_metrics_list
                for category, content in metrics_categories_dict_problem_level.items()
            },
            slot_set_dict={
                category: content.slot_set_metrics_list
                for category, content in metrics_categories_dict_problem_level.items()
            },
        ),
    )


def get_category_dict_list_from_problem_level_metrics(
    metrics_categories_dict_problem_level: Dict[
        str, List[MicroConfusionMetrixMetricsProblemLevelModel]
    ],
) -> Tuple[
    Dict[str, List[ConfusionMetrixMetricsModel]],
    Dict[str, List[ConfusionMetrixMetricsModel]],
    Dict[str, List[ConfusionMetrixMetricsModel]],
    Dict[str, List[ConfusionMetrixMetricsModel]],
]:
    intent_set_dict_list: Dict[str, List[ConfusionMetrixMetricsModel]] = dict()
    intent_counter_dict_list: Dict[str, List[ConfusionMetrixMetricsModel]] = dict()
    intent_list_dict_list: Dict[str, List[ConfusionMetrixMetricsModel]] = dict()
    slot_set_dict_list: Dict[str, List[ConfusionMetrixMetricsModel]] = dict()
    for category, metrics_models in metrics_categories_dict_problem_level.items():
        if category not in intent_set_dict_list:
            intent_set_dict_list[category] = []
            intent_counter_dict_list[category] = []
            intent_list_dict_list[category] = []
            slot_set_dict_list[category] = []
        for metrics_model in metrics_models:
            intent_set_dict_list[category].extend(metrics_model.intent_set_metrics_list)
            intent_counter_dict_list[category].extend(
                metrics_model.intent_counter_metrics_list
            )
            intent_list_dict_list[category].extend(
                metrics_model.intent_list_metrics_list
            )
            slot_set_dict_list[category].extend(metrics_model.slot_set_metrics_list)

    return (
        intent_set_dict_list,
        intent_counter_dict_list,
        intent_list_dict_list,
        slot_set_dict_list,
    )


def get_category_meta_metrics_aggregation_model(
    path_model_list: List[Tuple[Path, ScorerOuputModel]], categories: List[str]
) -> Tuple[MetaMetricsAggregationModel, MetaMetricsAggregationModel]:
    (
        metrics_categories_dict,
        metrics_categories_dict_problem_level,
        win_rate_categories_dict,
    ) = get_category_metrics_categories_dict(
        path_model_list=path_model_list, categories=categories
    )
    (
        intent_set_dict,
        intent_counter_dict,
        intent_list_dict,
        slot_set_dict,
    ) = get_meta_metrics_dict(metrics_categories_dict=metrics_categories_dict)
    (
        intent_set_dict_list,
        intent_counter_dict_list,
        intent_list_dict_list,
        slot_set_dict_list,
    ) = get_category_dict_list_from_problem_level_metrics(
        metrics_categories_dict_problem_level=metrics_categories_dict_problem_level
    )

    return (
        get_meta_metrics(
            intent_set_dict=intent_set_dict,
            intent_counter_dict=intent_counter_dict,
            intent_list_dict=intent_list_dict,
            slot_set_dict=slot_set_dict,
            win_rate_categories_dict=win_rate_categories_dict,
            categories=categories,
        ),
        get_meta_metrics(
            intent_set_dict=intent_set_dict_list,
            intent_counter_dict=intent_counter_dict_list,
            intent_list_dict=intent_list_dict_list,
            slot_set_dict=slot_set_dict_list,
        ),
    )
