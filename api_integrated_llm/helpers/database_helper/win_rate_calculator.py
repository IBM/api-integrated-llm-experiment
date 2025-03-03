import json
import os
from pathlib import Path
from typing import Optional

from api_integrated_llm.helpers.database_helper.metrics_helper.winrate_helper import (
    evaluate_win_rate,
    inference_call,
    setup,
)


def get_winrate(
    input_file: Path, db_path: Path, dataset_name: str = "superhero"
) -> Optional[float]:
    win_rate: Optional[float] = None
    # """

    # Usage:

    # PYTHONPATH=. python invocable_api_hub/driver/run_example.py --source_file sql_translation_output/dev/sequencing_bird_superhero.json -d superhero --database_directory db/dev_databases

    # Commmand line params:
    #     - source_file: input data
    #     - database_directory: location of database files
    #     - d: dataset name

    # """
    # input_file = args.source_file
    # db_path = args.database_directory
    # dataset_name = args.dataset
    builder, cache_file = setup(input_file, db_path)

    with open(input_file) as f:
        payloads = json.load(f)[
            "data"
        ]  # grab ['agent_data'] instead of ['data'] for react agent in tool-response_reflection

    for p in payloads:
        p["initialization_step"]["arguments"]["database_path"] = os.path.join(
            cache_file, dataset_name + ".sqlite"
        )

    with open("invocable_api_hub/driver/react_sql.txt", "r") as f:
        prompt_template = f.read().strip()

    # Dummy model inference results
    for p in payloads:
        model_output = inference_call(p, prompt_template, builder, "")
        p["model_output"] = model_output  # add model response here

    win_rate = evaluate_win_rate(payloads, builder)

    return win_rate
