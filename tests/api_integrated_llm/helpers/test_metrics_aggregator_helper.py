import os
from pathlib import Path
from api_integrated_llm.helpers.metrics_aggregator_helper import (
    get_metrics_aggregator_inputs,
)

project_root_path = Path(__file__).parent.parent.parent.parent.resolve()


def test_get_metrics_aggregator_inputs() -> None:
    metrics_objs, metrics_configuration_obj, has_error = get_metrics_aggregator_inputs(
        scoring_output_folder_path=Path(
            os.path.join(project_root_path, "tests", "data", "test_output", "scoring")
        ),
        metrics_aggregation_configuration_file_path=None,
    )

    assert not has_error
    assert len(metrics_objs) > 0
    assert len(metrics_configuration_obj) > 0
