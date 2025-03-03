from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from api_integrated_llm.data_models.source_models import (
    EvaluationOutputResponseDataUnit,
    QuerySourceModel,
)
from api_integrated_llm.helpers.database_helper.metrics_helper.win_rate_helper import (
    evaluate_win_rate,
    get_payloads_winrate,
    setup,
)
from api_integrated_llm.helpers.file_helper import get_base_model_from_json


def get_winrate(
    response_units: List[EvaluationOutputResponseDataUnit],
    source_file: Path,  # source data file
    db_path: Path,  # database path
    dataset_name: str = "superhero",  # dataset name
) -> Tuple[Optional[float], int, str]:
    win_rate: Optional[float] = None
    error_message: str = ""
    # """

    # Usage:

    # PYTHONPATH=. python invocable_api_hub/driver/run_example.py --source_file sql_translation_output/dev/sequencing_bird_superhero.json -d superhero --database_directory db/dev_databases

    # Commmand line params:
    #     - source_file: input data
    #     - database_directory: location of database files
    #     - d: dataset name

    # """
    try:
        builder, cache_file = setup(source_file, db_path)
        source_model: QuerySourceModel = get_base_model_from_json(
            file_path=source_file,
            base_model=QuerySourceModel,
        )
        payloads: List[Dict[str, Any]] = get_payloads_winrate(
            response_units=response_units,
            source_model=source_model,
            cache_file=cache_file,
            dataset_name=dataset_name,
        )

        if len(payloads) > 0:
            win_rate = evaluate_win_rate(payloads, builder)
    except Exception as e:
        error_message = str(e)

    return win_rate, len(payloads), error_message
