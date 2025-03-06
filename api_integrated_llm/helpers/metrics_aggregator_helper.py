from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

from api_integrated_llm.data_models.metrics_models import (
    DefaultMetricsAggregationConfiguration,
)
from api_integrated_llm.data_models.scorer_models import ScorerOuputModel
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
