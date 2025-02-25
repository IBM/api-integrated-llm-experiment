import json
from pathlib import Path
from typing import Any, Dict, List

from api_integrated_llm.data_models.source_models import (
    EvaluationOutputResponseDataUnit,
)


def get_evaluation_output_response_data_units_from_json(
    file_path: Path,
) -> List[EvaluationOutputResponseDataUnit]:
    objs: List[Dict[str, Any]] = []
    with open(file_path) as f:
        objs = json.load(f)

    outputs: List[EvaluationOutputResponseDataUnit] = []
    for obj in objs:
        outputs.append(
            EvaluationOutputResponseDataUnit(
                sample_id=obj["metadata"]["sample_id"],
                input=obj["input"],
                output=obj["output"],
                generated_text=json.dumps(obj["predicted_output"]),
                llm_model_id=obj["llm_data"]["llm_name"],
                source_file_path=obj["metadata"]["source_file_path"],
                dataset_name=obj["metadata"]["dataset"],
                temperature=obj["llm_data"]["temperature"],
                max_tokens=obj["llm_data"]["max_new_tokens"],
                is_agent=True,
            )
        )

    return outputs
