from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


from api_integrated_llm.data_models.source_models import (
    DataUnit,
    EvaluationOutputDataUnit,
    ExampleDataModel,
    QuerySourceModel,
)
from api_integrated_llm.helpers.file_helper import (
    get_base_model_from_json,
    get_dict_from_json,
    get_uuid4_str,
)
from api_integrated_llm.helpers.sampling_helper import get_random_example_for_prompt
from api_integrated_llm.helpers.tokenizer_helper import granite_prompt_input


def get_example_str(icl_examples: List[DataUnit], model_name: str) -> str:
    exampl_str = ""
    inputs = []
    output_fn_names = []
    idx = 1
    for ex in icl_examples:
        inputs.append(ex.input)
        output_fn_names.extend([f.name for f in ex.output])

        if model_name == "xLAM-7b-fc-r":
            exampl_str += f'\n#Example-{idx}\nInput: {ex.input}\nOutput: {{"tool_calls": {json.dumps(ex.output)} }}\n'
        elif model_name == "xLAM-1b-fc-r":
            exampl_str += f'\n#Example-{idx}\nInput: {ex.input}\nOutput: {{"tool_calls": {json.dumps(ex.output)} }}\n'
        elif model_name in ["xLAM-8x7b-r", "xLAM-8x22b-r"]:
            exampl_str += f'\n#Example-{idx}\nInput: {ex.input}\nOutput: {{"thought": "", "tool_calls": {json.dumps(ex.output)} }}\n'
        elif model_name == "Hermes-2-Pro-Mistral-7B":
            output_str = " ".join(
                [f"<tool_call> {json.dumps(f)} </tool_call>" for f in ex.output]
            )
            exampl_str += f"\n#Example-{idx}\nInput: {ex.input}\nOutput: {output_str}\n"
        else:
            exampl_str += f"\n#Example-{idx}\nInput: {ex.input}\nOutput: {json.dumps(ex.output)}\n"
        idx += 1
    return exampl_str


def sanitize_evaluation_input(json_dict: Dict[str, Any]) -> Dict[str, Any]:
    new_json_dict = deepcopy(json_dict)
    if "data" in json_dict:
        data = []
        for datum in json_dict["data"]:
            if ("ignore" in datum) and datum["ignore"]:
                continue
            json_str = json.dumps(datum)
            if "Union" not in json_str:
                data.append(deepcopy(datum))
        new_json_dict["data"] = data
    return new_json_dict


def is_inner_sourced_example_source(sample_dict: Dict[str, Any]) -> bool:
    return not (
        ("mathqa" in sample_dict)
        and ("stack" in sample_dict)
        and (len(sample_dict) == 2)
    )


def has_data(sample_dict: Dict[str, Any]) -> bool:
    return (
        False if ("data" not in sample_dict or len(sample_dict["data"]) == 0) else True
    )


def transform_to_example_source_from_inner_sourced_json(
    sample_dict: Dict[str, Any],
) -> Dict[str, Any]:
    if not has_data(sample_dict=sample_dict):
        raise Exception('"data" field is required to create an example source')

    sanitized_json_dict = sanitize_evaluation_input(json_dict=sample_dict)

    data = sanitized_json_dict["data"]
    dataset_name = data[0]["dataset_name"][:]

    return {dataset_name: deepcopy(data[:5])}


def is_inner_sourced_evaluation_source(file_path: Path) -> bool:
    return not str(file_path).endswith(".jsonl")


def transform_to_evaluation_source_from_inner_sourced_json(
    sample_dict: Dict[str, Any],
) -> List[Dict[str, Any]]:
    if not has_data(sample_dict=sample_dict):
        raise Exception('"data" field is required to create an evaluation source')

    sanitized_json_dict = sanitize_evaluation_input(json_dict=sample_dict)

    if len(sanitized_json_dict["data"]) == 0:
        raise Exception("No evaluation data is found.")

    return sanitized_json_dict["data"]


def get_examples(
    example_file_path: Path,
    evaluation_input_file_paths: List[Path],
    chosen_evaluation_input_file_path: Path,
    num_examples: int,
    should_generate_random_example: bool,
) -> List[DataUnit]:
    if should_generate_random_example:
        return get_random_example_for_prompt(
            evaluation_input_file_paths=evaluation_input_file_paths,  # type: ignore
            chosen_evaluation_input_file_path=chosen_evaluation_input_file_path,
            num_examples=num_examples,
        )

    example_model: ExampleDataModel = get_base_model_from_json(
        example_file_path, ExampleDataModel
    )

    return example_model.data


def get_prompt_dict(
    prompt_file_path: Path, evaluation_input_file_path: Path
) -> Dict[str, str]:
    prompt_dict_all = get_dict_from_json(file_path=prompt_file_path)  # type: ignore
    path_str = str(evaluation_input_file_path)
    if "rest" in path_str:
        return prompt_dict_all["router"]
    if "sequencing" in path_str:
        return prompt_dict_all["sequencing"]
    return prompt_dict_all["icl"]


def instruct_data(
    prompt_file_path: Path,
    model: str,
    evaluation_input_file_path: Path,
    evaluation_input_file_paths: List[str],
    example_file_path: Optional[Path] = None,
    should_generate_random_example: bool = False,
    num_examples: int = 1,
) -> List[EvaluationOutputDataUnit]:
    examples = get_examples(
        example_file_path=example_file_path,  # type: ignore
        evaluation_input_file_paths=evaluation_input_file_paths,  # type: ignore
        chosen_evaluation_input_file_path=evaluation_input_file_path,
        num_examples=num_examples,
        should_generate_random_example=should_generate_random_example,
    )

    if len(examples) == 0:
        raise Exception("No example data is found.")

    prompt_dict = get_prompt_dict(
        prompt_file_path=prompt_file_path,
        evaluation_input_file_path=evaluation_input_file_path,
    )

    source_model: QuerySourceModel = (
        get_base_model_from_json(
            file_path=evaluation_input_file_path,
            base_model=QuerySourceModel,
        ),
    )

    test_data: List[EvaluationOutputDataUnit] = []
    example_str = get_example_str(examples, model)
    for sample in source_model.data:
        function_str = json.dumps(
            list((map(lambda item: item.model_dump(), sample.tools)))
        )
        key_value_description_str = json.dumps(
            list(
                (
                    map(
                        lambda item: item.model_dump(),
                        sample.key_values_and_descriptions,
                    )
                )
            )
        )
        if "granite" in model.lower():
            input_prompt = granite_prompt_input(
                sample.input,
                sample.tools,
                example_str,
                prompt_dict["granite"],
                key_value_description_str,
            )
        elif "llama" in model.lower():
            input_prompt = prompt_dict["LLaMa-3.1"].format(
                FUNCTION_STR=function_str,
                ICL_EXAMPLES=example_str,
                QUERY=sample.input,
                KEY_VALUES_AND_DESCRIPTIONS=key_value_description_str,
            )
        else:
            try:
                tmp_key = model[:]
                if tmp_key not in prompt_dict:  # handle exceptions
                    tmp_key = "llama-3-1-405b-instruct"

                input_prompt = prompt_dict[tmp_key].format(
                    FUNCTION_STR=function_str,
                    ICL_EXAMPLES=example_str,
                    QUERY=sample.input,
                    KEY_VALUES_AND_DESCRIPTIONS=key_value_description_str,
                )
            except:
                input_prompt = (
                    prompt_dict[model]
                    .replace("{FUNCTION_STR}", function_str)
                    .replace("{ICL_EXAMPLES}", example_str)
                    .replace("{QUERY}", sample.input)
                    .replace("{KEY_VALUES_AND_DESCRIPTIONS}", key_value_description_str)
                )

        test_data.append(
            EvaluationOutputDataUnit(
                sample_id=sample.sample_id,
                input=input_prompt,
                output=sample.output,
                gold_answer=sample.gold_answer,
            )
        )

    return test_data
