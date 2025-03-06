from pathlib import Path
from typing import Optional

from api_integrated_llm.helpers.metrics_aggregator_helper import (
    get_metrics_aggregator_inputs,
)


def aggregate_metrics(
    scoring_output_folder_path: Optional[Path],
    metrics_aggregation_configuration_file_path: Optional[Path],
) -> bool:
    metrics_objs, metrics_configuration_obj, has_error = get_metrics_aggregator_inputs(
        scoring_output_folder_path=scoring_output_folder_path,
        metrics_aggregation_configuration_file_path=metrics_aggregation_configuration_file_path,
    )

    if has_error:
        return has_error

    return has_error
