import asyncio
from copy import deepcopy
from pathlib import Path
import os
from typing import Dict, List

from api_integrated_llm.data_models.source_models import (
    EvaluationOutputDataUnit,
    EvaluationOutputResponseDataUnit,
)
from api_integrated_llm.helpers.file_helper import (
    get_file_name_without_extension,
    write_json_from_dict,
    write_jsonl,
)
from api_integrated_llm.helpers.service_helper import (
    get_responses_from_async,
)
from api_integrated_llm.helpers.instruct_data_prep import instruct_data

loop = asyncio.get_event_loop()


def get_evaluation_output_units_from_responses(
    model_name: str,
    test_data: List[EvaluationOutputDataUnit],
    responses: List[str],
    evaluation_input_file_path: Path,
    dataset_name: str,
    temperature: float,
    max_tokens: int,
) -> List[EvaluationOutputResponseDataUnit]:
    output_list: List[EvaluationOutputResponseDataUnit] = []
    for sample, resp in zip(test_data, responses):
        if resp is not None and isinstance(resp, list) and len(resp) > 0:
            output_unit = EvaluationOutputResponseDataUnit.get_model_from_output_unit(
                data_model=sample
            )
            output_unit.generated_text = resp[0].strip()
            output_unit.llm_model_id = model_name[:]
            output_unit.source_file_path = str(evaluation_input_file_path)
            output_unit.dataset_name = dataset_name
            output_unit.temperature = temperature
            output_unit.max_tokens = max_tokens
            output_list.append(output_unit)
    return output_list


async def get_output_list(
    prompt_file_path: Path,
    evaluation_input_file_path: Path,
    evaluation_input_file_paths: List[Path],
    example_file_path: Path,
    error_folder_path: Path,
    output_folder_path: Path,
    model_name: str,
    dataset_name: str,
    should_generate_random_example: bool,
    num_examples: int,
    should_ignore: bool,
    model_obj,
    temperature: float,
    max_tokens: int,
    temperature_str: str,
    max_tokens_str: str,
) -> None:
    output_list: List[EvaluationOutputResponseDataUnit] = []
    try:
        test_data = instruct_data(
            prompt_file_path=prompt_file_path,
            model_name=model_name,
            evaluation_input_file_path=evaluation_input_file_path,  # type: ignore
            evaluation_input_file_paths=evaluation_input_file_paths,
            example_file_path=example_file_path,
            should_generate_random_example=should_generate_random_example,
            num_examples=num_examples,
            should_ignore=should_ignore,
        )

        if model_obj["inference_type"] == "RITS":
            responses = await get_responses_from_async(
                test_data=test_data,
                model_obj=model_obj,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            output_list.extend(
                get_evaluation_output_units_from_responses(
                    model_name=model_name,
                    test_data=test_data,
                    responses=responses,
                    evaluation_input_file_path=evaluation_input_file_path,
                    dataset_name=dataset_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
        else:
            raise Exception(
                "Model inference type does not match existing implementations"
            )
    except Exception as e:
        print(e)
        write_json_from_dict(
            file_path=Path(
                os.path.join(
                    error_folder_path,
                    model_name,
                    temperature_str,
                    max_tokens_str,
                    dataset_name + "_evaluation" + ".json",
                )
            ),
            dic={"error": str(e)},
        )

    write_jsonl(
        file_path=Path(
            os.path.join(
                output_folder_path,
                model_name,
                temperature_str,
                max_tokens_str,
                dataset_name + ".jsonl",
            )
        ),
        jsons=output_list,
    )


def evaluate(
    model_id_info_dict: Dict[str, Dict[str, str]],
    evaluation_input_file_paths: List[Path],
    example_file_path: Path,
    output_folder_path: Path,
    prompt_file_path: Path,
    error_folder_path: Path,
    temperatures: List[float],
    max_tokens_list: List[int],
    should_generate_random_example: bool = False,
    num_examples: int = 1,
    should_ignore: bool = True,
):
    for temperature in temperatures:
        print(f"Temperature: {temperature}")
        temperature_str = "temperature_" + str(temperature).replace(".", "_")
        for max_tokens in max_tokens_list:
            print(f"Max tokens: {max_tokens}")
            max_tokens_str = "maxtokens_" + str(max_tokens)

            for evaluation_input_file_path in evaluation_input_file_paths:
                tasks = []
                for model_name, model_obj in model_id_info_dict.items():
                    dataset_name = get_file_name_without_extension(
                        file_path=evaluation_input_file_path  # type: ignore
                    )
                    tasks.append(
                        get_output_list(
                            prompt_file_path=deepcopy(prompt_file_path),
                            evaluation_input_file_path=deepcopy(
                                evaluation_input_file_path
                            ),
                            evaluation_input_file_paths=deepcopy(
                                evaluation_input_file_paths
                            ),
                            example_file_path=deepcopy(example_file_path),
                            error_folder_path=deepcopy(error_folder_path),
                            output_folder_path=deepcopy(output_folder_path),
                            model_name=model_name[:],
                            dataset_name=dataset_name[:],
                            should_generate_random_example=should_generate_random_example,
                            num_examples=num_examples,
                            should_ignore=should_ignore,
                            model_obj=deepcopy(model_obj),
                            temperature=temperature,
                            max_tokens=max_tokens,
                            temperature_str=temperature_str[:],
                            max_tokens_str=max_tokens_str[:],
                        )
                    )

                loop.run_until_complete(asyncio.gather(*tasks))
