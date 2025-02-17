from copy import deepcopy
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Set

from api_integrated_llm.data_models.core_data_models import DataUnit
from api_integrated_llm.helpers.file_helper import get_dict_from_json


def get_random_example_for_prompt(
    evaluation_input_file_paths: List[str],
    chosen_evaluation_input_file_path: Path,
    num_examples: int,
) -> Dict[str, Dict[str, Any]]:
    if len(evaluation_input_file_paths) <= 1:
        return {}
    sampled_examples: List[Dict[str, Any]] = []
    sample_hash_to_skip: Set[int] = set()
    max_try = 10000
    num_try = 0
    while len(sampled_examples) < num_examples and num_try < max_try:
        sample_file_idx = random.randint(0, len(evaluation_input_file_paths) - 1)
        file_to_sample = evaluation_input_file_paths[sample_file_idx]
        if file_to_sample == chosen_evaluation_input_file_path:
            num_try += 1
            continue

        parsed_sample = get_dict_from_json(file_path=file_to_sample)
        if "data" not in parsed_sample:
            num_try += 1
            continue

        data_pool = parsed_sample["data"]
        if len(data_pool) == 0:
            num_try += 1
            continue

        sample_point_idx = random.randint(0, len(data_pool) - 1)
        sample = data_pool[sample_point_idx]
        sample_hash = hash(json.dumps(sample, sort_keys=True))
        if sample_hash in sample_hash_to_skip:
            num_try += 1
            continue

        try:
            sampled_examples.append(
                DataUnit(
                    input=sample["input"][:],
                    output=deepcopy(sample["output"]),
                    tools=deepcopy(sample["tools"]),
                    gold_answer=deepcopy(sample["gold_answer"]),
                ).dict()
            )
        except Exception as e:
            print(e)
            num_try += 1
            continue
        sample_hash_to_skip.add(sample_hash)

    return {"random_examples": sampled_examples}
