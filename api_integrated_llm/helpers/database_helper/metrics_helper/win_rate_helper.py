from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from api_integrated_llm.data_models.scorer_models import (
    WinRateResultModel,
    WinRateResultUnitModel,
)
from api_integrated_llm.data_models.source_models import (
    QuerySourceModel,
)
from api_integrated_llm.helpers.database_helper.database_builders.sql_dataset_builder import (
    SqlDatasetBuilder,
)
from api_integrated_llm.helpers.database_helper.database_builders.sql_sequencing_dataset_builder import (
    SqlSequencingDatasetBuilder,
)
from api_integrated_llm.helpers.database_helper.database_builders.sql_slot_filling_dataset_builder import (
    SqlSlotFillingDatasetBuilder,
)
from api_integrated_llm.helpers.database_helper.core_components.driver_components import (
    validate_api_output,
    check_equality_without_order,
)
from api_integrated_llm.helpers.file_helper import get_json_dict_from_txt, get_uuid4_str


def setup(input_file: str, db_path: str):
    # cache_path is where a temporary copy of the db will be located
    # This is where we write temp tables from joins, and with cleaned table/column names
    # Currently this path needs to be an absolute path.
    cache_path = os.path.join(db_path, "cache")
    os.makedirs(cache_path, exist_ok=True)

    if not os.path.isabs(cache_path):
        cache_path = os.path.join(os.getcwd(), cache_path)

    assert os.path.isfile(input_file), f"Missing file {input_file}"
    filename = os.path.basename(input_file)

    if "_sparc_" in filename:
        source = "sparc"
        database_name = filename.split("_sparc_")[1].replace(".json", "")
    elif "_bird_" in filename:
        source = "bird"
        database_name = filename.split("_bird_")[1].replace(".json", "")
    else:
        raise Exception("Bad file")

    if filename.startswith("sequencing"):
        builder = SqlSequencingDatasetBuilder(
            database_name, db_path, cache_path, source_dataset_name=source
        )
    elif filename.startswith("slot_filling"):
        builder = SqlSlotFillingDatasetBuilder(
            database_name, db_path, cache_path, source_dataset_name=source
        )
    else:
        raise Exception("Bad file")
    builder.build()

    return builder, cache_path


def inference_call(
    payload: dict, prompt_template: str, builder: SqlDatasetBuilder, runnable
):
    api_pool, _ = builder.set_query_specific_api_pool([payload["initialization_step"]])

    API_name_list = []
    api_descriptions = {}

    # Format the tool specifications part of the prompt
    tools_info = payload["tools"]
    for tool_spec in tools_info:
        tool_name = tool_spec["name"]
        api_descriptions[tool_name] = json.dumps(tool_spec)
        API_name_list.append(tool_name)

    # Get the extra data describing the valid key_names parameters (column names)
    # key_names_and_descriptions = payload["key_values_and_descriptions"]
    # key_names_and_desc_str = "\n".join(
    #     [k["key_name"] + ": " + k["description"] for k in key_names_and_descriptions]
    # )
    # key_names_str = ", ".join([k["key_name"] for k in key_names_and_descriptions])

    # Fill in the prompt template
    # prompt_str = prompt_template.format(
    #     tools=api_descriptions,
    #     tool_names="\n".join(API_name_list),
    #     input=payload["input"],
    #     previousruns="",
    #     agent_scratchpad="",
    #     key_enum=key_names_and_desc_str,
    #     key_names=key_names_str,
    # )
    # conversation = [
    #     {"content": "you are a helpful assistant", "role": "system"},
    #     {"content": prompt_str, "role": "user"},
    # ]

    agent_trajectory = []

    first_tao_step = {}
    # Run the initialization step by hand. If running a tao loop, fill in the inital loop with the data point init function like so:
    first_tao_step["action"] = payload["initialization_step"]["name"]
    first_tao_step["action_input"] = payload["initialization_step"]["arguments"]
    first_tao_step[
        "model_response"
    ] = f"Thought: I need to get the data first, Action: {first_tao_step['action']}, Action Input: {first_tao_step['action_input']}"
    agent_trajectory.append(first_tao_step)

    for i in range(5):  # Loop over tao iterations
        # Stub
        # response = runnable.invoke(conversation)
        response = {}

        # The model response should include a choice of api name ("action") and the associated parameters ("action_input")
        # the chosen api must be available in api_pool
        chosen_api = api_pool.get(response.get("action"), None)
        if chosen_api is None:
            observation = ""  # Stub for now, but chosen_api must be in the api_pool to be a valid choice
        else:
            observation = chosen_api(**response.get("action_input"))
        response["API_response"] = observation

        agent_trajectory.append(response)
    return agent_trajectory


def get_repaired_function_calls(
    required_api_calls: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    repaired_function_calls = []
    for function_call in required_api_calls:
        call = deepcopy(function_call)
        if "name" not in call:
            call["name"] = get_uuid4_str()
        if "arguments" not in call:
            call["arguments"] = {}
        repaired_function_calls.append(call)
    return repaired_function_calls


def evaluate_win_rate(
    payloads: List[Dict[str, Any]],
    builder: SqlDatasetBuilder,
    pred_function_calls_list: List[List[Any]],
    gold_function_calls_list: List[List[Any]],
) -> Tuple[float, List[str], int, WinRateResultModel]:
    valid = []
    win_rate_result_model: WinRateResultModel = WinRateResultModel()
    num_failed_function_execution_tot = 0
    error_messages_tot: List[str] = []
    for i, p in enumerate(payloads):
        try:
            # Set the database path to the cache file of the initialized builder.
            # Otherwise it will point to the cache location from the data generation run.
            p["initialization_step"]["arguments"][
                "database_path"
            ] = builder.loader.cache_file

            if "arguments" not in p:
                p["arguments"] = {}  # handle alternative payload

            # If we used the original output sequence here, instead of the model output, it would just check the correctness of the data point
            required_api_calls = [p["initialization_step"]]
            required_api_calls.extend(
                p["output"]
            )  # ie. change 'model_output' to just 'output', and the win_rate should be 1.0
            required_api_calls = get_repaired_function_calls(
                required_api_calls=required_api_calls
            )
            # This dictionary has [key, value] == [tool_name, tool (executable python function)]
            # It is needed for evaluating the win rate, it is NOT consumed by the tool calling model
            api_pool, _ = builder.set_query_specific_api_pool(
                [p["initialization_step"]]
            )

            (
                api_result,
                error_messages,
                num_failed_function_execution,
            ) = validate_api_output(required_api_calls, api_pool)
            error_messages_tot.extend(error_messages)
        except:
            num_failed_function_execution = len(p["output"])
            api_result = None

        num_failed_function_execution_tot + num_failed_function_execution
        validated = check_equality_without_order(api_result, p["gold_answer"])
        valid.append(validated)
        win_rate_result_model.win_rate_result.append(
            WinRateResultUnitModel(
                valid=validated,
                pred_function_calls=deepcopy(pred_function_calls_list[i]),
                gold_function_calls=deepcopy(gold_function_calls_list[i]),
                num_failed_function_execution=num_failed_function_execution,
                error_messages=deepcopy(error_messages),
            )
        )

    return (
        (sum(valid) / len(valid)),
        error_messages_tot,
        num_failed_function_execution_tot,
        win_rate_result_model,
    )


def parse_sequence(function_list: List[Any]) -> Tuple[List[Dict[str, Any]], List[str]]:
    function_dict_list: List[Dict[str, Any]] = []
    error_messages: List[str] = []
    for content in function_list:
        json_dict: Optional[Dict[str, Any]] = None
        try:
            if isinstance(content, str):
                parsed_content = get_json_dict_from_txt(txt=content)
                if not isinstance(parsed_content, dict):
                    error_messages.append("Parsed function is not list: {content}")
                    continue
                json_dict = parsed_content
            elif isinstance(content, dict):
                json_dict = content
        except Exception as e:
            error_messages.append(f"Exception: {str(e)} \n content: {content}")

        if json_dict is not None:
            function_dict_list.append(json_dict)
    return function_dict_list, error_messages


def get_payloads_winrate(
    source_model: QuerySourceModel,
    cache_folder_path: Path,
    dataset_name: str,
    predicted_function_calls_tuple: List[Tuple[Union[str, int], List[Any]]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    error_messages: List[str] = []
    sample_id_predicted_function_calls_dict = {
        str(sample_id): deepcopy(predicted_function_calls)
        for sample_id, predicted_function_calls in predicted_function_calls_tuple
    }

    payloads: List[Dict[str, Any]] = []
    for datum in source_model.data:
        sample_id_str = str(datum.sample_id)
        if sample_id_str in sample_id_predicted_function_calls_dict:
            try:
                sequence, error_messages_instance = parse_sequence(
                    function_list=sample_id_predicted_function_calls_dict[sample_id_str]
                )
                error_messages.extend(error_messages_instance)
                if len(error_messages_instance) == 0:
                    payload = datum.model_dump()
                    payload["initialization_step"]["arguments"][
                        "database_path"
                    ] = os.path.join(cache_folder_path, dataset_name + ".sqlite")
                    payload["output"] = sequence
                    payloads.append(payload)
            except Exception as e:
                error_messages.append(
                    f"Exception thrown with {sample_id_str}: {str(e)}"
                )
    return payloads, error_messages
