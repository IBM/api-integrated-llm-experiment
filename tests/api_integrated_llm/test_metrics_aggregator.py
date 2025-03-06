import os
from pathlib import Path
from api_integrated_llm.helpers.file_helper import get_date_time_str
from api_integrated_llm.metrics_aggregator import aggregate_metrics


project_root_path = Path(__file__).parent.parent.parent.resolve()


def test_aggregate_metrics() -> None:
    time_str = get_date_time_str()
    has_error = aggregate_metrics(
        metrics_aggregator_input_path=Path(
            os.path.join(
                project_root_path,
                "tests",
                "data",
                "test_output",
                "scoring",
            )
        ),
        metrics_aggregator_output_file_path=Path(
            os.path.join(
                project_root_path,
                "output",
                "metrics_aggregation",
                f"metrics_aggregation_{time_str}.json",
            )
        ),
        metrics_aggregation_configuration_file_path=None,
    )

    assert not has_error
