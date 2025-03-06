from pathlib import Path
from typing import Dict, Optional

from api_integrated_llm.data_models.scorer_models import (
    AggegatorOutputModel,
    MetaMetricsAggregationModel,
)
from api_integrated_llm.helpers.file_helper import write_json
from api_integrated_llm.helpers.metrics_aggregator_helper import (
    get_agent_meta_metrics_aggregation_model,
    get_category_meta_metrics_aggregation_model,
    get_metrics_aggregator_inputs,
)


def aggregate_metrics(
    metrics_aggregator_input_path: Path,
    metrics_aggregator_output_path: Path,
    metrics_aggregation_configuration_file_path: Optional[Path],
) -> bool:
    metrics_objs, metrics_configuration_obj, has_error = get_metrics_aggregator_inputs(
        scoring_output_folder_path=metrics_aggregator_input_path,
        metrics_aggregation_configuration_file_path=metrics_aggregation_configuration_file_path,
    )

    if has_error:
        return has_error

    try:
        aggregated_metrics: Dict[str, MetaMetricsAggregationModel] = {
            "compute_mode_meta_metrics": get_agent_meta_metrics_aggregation_model(
                path_model_list=metrics_objs,
            )
        }
        for category_mode, categoris in metrics_configuration_obj.items():
            aggregated_metrics[
                category_mode
            ] = get_category_meta_metrics_aggregation_model(
                path_model_list=metrics_objs, categories=categoris
            )

        write_json(
            file_path=metrics_aggregator_output_path,
            base_model=AggegatorOutputModel(aggregate_metrics=aggregated_metrics),
        )
    except Exception as e:
        has_error = True
        print(f"Error at Metrics Aggregator: {str(e)}")

    return has_error
