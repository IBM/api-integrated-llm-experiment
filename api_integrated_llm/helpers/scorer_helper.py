import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from api_integrated_llm.data_models.auxiliary_models import SampleIgonoreModel
from api_integrated_llm.data_models.source_models import (
    EvaluationOutputResponseDataUnit,
)
from api_integrated_llm.helpers.file_helper import (
    get_base_models_from_jsonl,
    get_dict_from_json,
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


def get_sample_ignore_model_from_file(
    ignore_file_path: Optional[Path],
) -> Optional[SampleIgonoreModel]:
    sample_ignore_model: Optional[SampleIgonoreModel] = None
    if ignore_file_path is not None:
        try:
            sample_ignore_model = SampleIgonoreModel.get_sample_ignore_model(
                sample_dict=get_dict_from_json(file_path=ignore_file_path)
            )
        except Exception as e:
            error_message = (
                f"Error at constructing SampleIgnoreModel at Scorer: {str(e)}"
            )
            print(error_message)
            raise Exception(error_message)
    return sample_ignore_model


def get_evaluation_output_response_data_units(
    evaluator_output_file_path: Path, sample_ignore_model: Optional[SampleIgonoreModel]
) -> Tuple[List[EvaluationOutputResponseDataUnit], int]:
    data_units: List[EvaluationOutputResponseDataUnit] = (
        get_base_models_from_jsonl(
            file_path=evaluator_output_file_path,
            base_model=EvaluationOutputResponseDataUnit,
        )
        if str(evaluator_output_file_path).endswith("jsonl")
        else get_evaluation_output_response_data_units_from_json(
            file_path=evaluator_output_file_path,
        )
    )
    filtered_data_units: List[EvaluationOutputResponseDataUnit] = []
    num_samples_ignored: int = 0

    if sample_ignore_model is not None:
        for data_unit in data_units:
            if data_unit.sample_id is not None:
                if sample_ignore_model.has_sample_to_ignore(
                    file_path=data_unit.source_file_path, sample_id=data_unit.sample_id
                ):
                    num_samples_ignored += 1
                    continue

            filtered_data_units.append(data_unit.model_copy(deep=True))
    else:
        filtered_data_units = data_units

    return filtered_data_units, num_samples_ignored
