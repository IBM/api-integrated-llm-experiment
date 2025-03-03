import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from api_integrated_llm.data_models.source_models import (
    EvaluationOutputResponseDataUnit,
    QuerySourceModel,
)
from api_integrated_llm.helpers.database_helper.metrics_helper.win_rate_helper import (
    evaluate_win_rate,
    setup,
)
from api_integrated_llm.helpers.file_helper import get_base_model_from_json


def get_winrate(
    response_units: List[EvaluationOutputResponseDataUnit],
    source_file: Path,
    db_path: Path,
    dataset_name: str = "superhero",
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
        sample_id_response_dict = {
            response_unit.sample_id: response_unit.generated_text
            for response_unit in response_units
        }

        for datum in source_model.data:
            datum.initialization_step = {
                "arguments": {
                    "database_path": os.path.join(cache_file, dataset_name + ".sqlite")
                }
            }

        payloads: List[Dict[str, Any]] = []
        for datum in source_model.data:
            if datum.sample_id in sample_id_response_dict:
                payload = datum.model_dump()
                payload["model_output"] = sample_id_response_dict[payload["sample_id"]]
                payload["initialization_step"]["arguments"][
                    "database_path"
                ] = os.path.join(cache_file, dataset_name + ".sqlite")
                payloads.append(payload)
        if len(payloads) > 0:
            win_rate = evaluate_win_rate(payloads, builder)
    except Exception as e:
        error_message = str(e)

    return win_rate, len(payloads), error_message
