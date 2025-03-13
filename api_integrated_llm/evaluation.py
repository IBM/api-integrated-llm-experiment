import asyncio
from copy import deepcopy
from pathlib import Path
import os
from typing import Any, Dict, List, Optional, Union

from api_integrated_llm.data_models.source_models import (
    EvaluationOutputDataUnit,
    EvaluationOutputResponseDataUnit,
)
from api_integrated_llm.helpers.database_helper.local_llm_helper import (
    get_responses_from_local_llm,
)
from api_integrated_llm.helpers.file_helper import (
    get_uuid4_str,
    write_json_from_dict,
    write_jsonl,
)
from api_integrated_llm.helpers.service_helper import (
    get_responses_from_async,
    get_responses_from_sync,
)
from api_integrated_llm.helpers.instruct_data_prep import instruct_data


def get_evaluation_output_units_from_responses(
    model_name: str,
    test_data: List[EvaluationOutputDataUnit],
    responses: List[Union[List[str], str, None]],
    evaluation_input_file_path: Path,
    dataset_name: str,
    temperature: float,
    max_tokens: int,
) -> List[EvaluationOutputResponseDataUnit]:
    output_list: List[EvaluationOutputResponseDataUnit] = []
    for sample, resp in zip(test_data, responses):
        if resp is not None:
            response = ""
            if isinstance(resp, list) and len(resp) > 0:
                try:
                    response = resp[0].strip()
                except Exception as e:
                    print(e)
            if isinstance(response, str):
                response = resp.strip()  # type: ignore

            output_unit = EvaluationOutputResponseDataUnit.get_model_from_output_unit(
                data_model=sample
            )
            output_unit.generated_text = response
            output_unit.llm_model_id = model_name[:]
            output_unit.source_file_path = str(evaluation_input_file_path)
            output_unit.dataset_name = dataset_name
            output_unit.temperature = temperature
            output_unit.max_tokens = max_tokens
            output_list.append(output_unit)
    return output_list


async def get_output_list_async(
    prompt_file_path: Path,
    evaluation_input_file_path: Path,
    evaluation_input_file_paths: List[Path],
    example_file_path: Path,
    error_folder_path: Path,
    output_folder_path: Path,
    model_name: str,
    should_generate_random_example: bool,
    num_examples: int,
    should_ignore: bool,
    model_obj,
    temperature: float,
    max_tokens: int,
) -> None:
    temperature_str = f"temperature_{temperature}"
    max_tokens_str = f"maxtoken_{max_tokens}"
    agent_str = "llm"
    output_list: List[EvaluationOutputResponseDataUnit] = []
    output_file_name = str(evaluation_input_file_path).split("/")[-1].split(".")[0]
    hash_str = get_uuid4_str()

    try:
        test_data, dataset = instruct_data(
            prompt_file_path=prompt_file_path,
            model_name=model_name,
            evaluation_input_file_path=evaluation_input_file_path,  # type: ignore
            evaluation_input_file_paths=evaluation_input_file_paths,
            example_file_path=example_file_path,
            should_generate_random_example=should_generate_random_example,
            num_examples=num_examples,
            should_ignore=should_ignore,
        )
        if len(test_data) > 0:
            responses: List[Optional[Union[List[str], str]]] = []
            if model_obj["endpoint"].startswith("http"):
                responses = await get_responses_from_async(
                    test_data=test_data,
                    model_obj=model_obj,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                raise Exception(
                    "Local LLM use for async operations has not been implemented."
                )

            output_list.extend(
                get_evaluation_output_units_from_responses(
                    model_name=model_name.split("/")[-1],
                    test_data=test_data,
                    responses=responses,
                    evaluation_input_file_path=evaluation_input_file_path,
                    dataset_name=(
                        dataset
                        if dataset is not None
                        else f"default_dataset_{hash_str}"
                    ),
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
                    output_file_name + "_" + hash_str + ".json",
                )
            ),
            dic={"error": str(e)},
        )

    if len(output_list) > 0:
        temperature_str, max_tokens_str, _, model_name, agent_str = output_list[
            0
        ].get_basic_strs()

    write_jsonl(
        file_path=Path(
            os.path.join(
                output_folder_path,
                agent_str,
                model_name,
                temperature_str,
                max_tokens_str,
                output_file_name + ".jsonl",
            )
        ),
        jsons=output_list,
    )


def get_output_list(
    prompt_file_path: Path,
    evaluation_input_file_path: Path,
    evaluation_input_file_paths: List[Path],
    example_file_path: Path,
    error_folder_path: Path,
    output_folder_path: Path,
    model_name: str,
    should_generate_random_example: bool,
    num_examples: int,
    should_ignore: bool,
    model_obj,
    temperature: float,
    max_tokens: int,
) -> None:
    temperature_str = f"temperature_{temperature}"
    max_tokens_str = f"maxtoken_{max_tokens}"
    agent_str = "llm"
    output_list: List[EvaluationOutputResponseDataUnit] = []
    output_file_name = str(evaluation_input_file_path).split("/")[-1].split(".")[0]
    hash_str = get_uuid4_str()

    try:
        test_data, dataset = instruct_data(
            prompt_file_path=prompt_file_path,
            model_name=model_name,
            evaluation_input_file_path=evaluation_input_file_path,  # type: ignore
            evaluation_input_file_paths=evaluation_input_file_paths,
            example_file_path=example_file_path,
            should_generate_random_example=should_generate_random_example,
            num_examples=num_examples,
            should_ignore=should_ignore,
        )
        if len(test_data) > 0:
            responses: List[str] = []
            if model_obj["endpoint"].startswith("http"):
                responses = get_responses_from_sync(
                    test_data=test_data,
                    model_obj=model_obj,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            else:
                responses, error_messages = get_responses_from_local_llm(
                    test_data=test_data,
                    model_obj=model_obj,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                if len(error_messages) > 0:
                    write_json_from_dict(
                        file_path=Path(
                            os.path.join(
                                error_folder_path,
                                model_name,
                                temperature_str,
                                max_tokens_str,
                                output_file_name + "_" + hash_str + ".json",
                            )
                        ),
                        dic={"error": error_messages},
                    )

            output_list.extend(
                get_evaluation_output_units_from_responses(
                    model_name=model_name.split("/")[-1],
                    test_data=test_data,
                    responses=responses,  # type: ignore
                    evaluation_input_file_path=evaluation_input_file_path,
                    dataset_name=(
                        dataset
                        if dataset is not None
                        else f"default_dataset_{hash_str}"
                    ),
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
                    output_file_name + "_" + hash_str + ".json",
                )
            ),
            dic={"error": str(e)},
        )

    if len(output_list) > 0:
        temperature_str, max_tokens_str, _, model_name, agent_str = output_list[
            0
        ].get_basic_strs()

    write_jsonl(
        file_path=Path(
            os.path.join(
                output_folder_path,
                agent_str,
                model_name,
                temperature_str,
                max_tokens_str,
                output_file_name + ".jsonl",
            )
        ),
        jsons=output_list,
    )


def evaluate(
    model_id_info_dict: Dict[str, Dict[str, Any]],
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
    should_async: bool = True,
) -> None:
    loop = asyncio.get_event_loop()

    for temperature in temperatures:
        print(f"Temperature: {temperature}")

        for max_tokens in max_tokens_list:
            print(f"Max tokens: {max_tokens}")

            for evaluation_input_file_path in evaluation_input_file_paths:
                if should_async:
                    tasks = []  # type: ignore
                    for model_name, model_obj in model_id_info_dict.items():
                        tasks.append(
                            get_output_list_async(
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
                                should_generate_random_example=should_generate_random_example,
                                num_examples=num_examples,
                                should_ignore=should_ignore,
                                model_obj=deepcopy(model_obj),
                                temperature=temperature,
                                max_tokens=max_tokens,
                            )
                        )

                    loop.run_until_complete(asyncio.gather(*tasks))
                else:
                    for model_name, model_obj in model_id_info_dict.items():
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
                            should_generate_random_example=should_generate_random_example,
                            num_examples=num_examples,
                            should_ignore=should_ignore,
                            model_obj=deepcopy(model_obj),
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
