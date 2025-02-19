import json
from pathlib import Path
import random
from typing import List, Set

from api_integrated_llm.data_models.source_models import DataUnit, QuerySourceModel
from api_integrated_llm.helpers.file_helper import (
    get_base_model_from_json,
)


def get_random_example_for_prompt(
    evaluation_input_file_paths: List[str],
    chosen_evaluation_input_file_path: Path,
    num_examples: int,
) -> List[DataUnit]:
    if len(evaluation_input_file_paths) <= 1:
        return []
    sampled_examples: List[DataUnit] = []
    sample_hash_to_skip: Set[int] = set()
    max_try = 10000
    num_try = 0
    while len(sampled_examples) < num_examples and num_try < max_try:
        sample_file_idx = random.randint(0, len(evaluation_input_file_paths) - 1)
        file_to_sample = evaluation_input_file_paths[sample_file_idx]
        if file_to_sample == chosen_evaluation_input_file_path:
            num_try += 1
            continue

        source_model: QuerySourceModel = get_base_model_from_json(
            file_path=Path(file_to_sample),
            base_model=QuerySourceModel,
        )
        if source_model.data is None or len(source_model.data) == 0:
            num_try += 1
            continue

        sample_point_idx = random.randint(0, len(source_model.data) - 1)
        sample = source_model.data[sample_point_idx]
        sample_hash = hash(json.dumps(sample.model_dump(), sort_keys=True))
        if sample_hash in sample_hash_to_skip:
            num_try += 1
            continue

        try:
            sampled_examples.append(sample.get_data_unit_model())
        except Exception as e:
            print(e)
            num_try += 1
            continue
        sample_hash_to_skip.add(sample_hash)

    return sampled_examples
