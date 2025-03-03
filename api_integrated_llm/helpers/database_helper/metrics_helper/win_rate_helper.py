import json
import os
from typing import Any, Dict, List

from api_integrated_llm.data_models.source_models import (
    EvaluationOutputResponseDataUnit,
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


def evaluate_win_rate(payloads: list[dict], builder: SqlDatasetBuilder):
    valid = []
    for p in payloads:
        # Set the database path to the cache file of the initialized builder.
        # Otherwise it will point to the cache location from the data generation run.
        p["initialization_step"]["arguments"][
            "database_path"
        ] = builder.loader.cache_file

        # If we used the original output sequence here, instead of the model output, it would just check the correctness of the data point
        required_api_calls = [p["initialization_step"]]
        required_api_calls.extend(
            p["output"]
        )  # ie. change 'model_output' to just 'output', and the win_rate should be 1.0

        # This dictionary has [key, value] == [tool_name, tool (executable python function)]
        # It is needed for evaluating the win rate, it is NOT consumed by the tool calling model
        api_pool, _ = builder.set_query_specific_api_pool([p["initialization_step"]])

        api_result = validate_api_output(required_api_calls, api_pool)
        validated = check_equality_without_order(api_result, p["gold_answer"])
        valid.append(validated)

    return sum(valid) / len(valid)


def get_payloads_winrate(
    response_units: List[EvaluationOutputResponseDataUnit],
    source_model: QuerySourceModel,
    cache_file: Any,
    dataset_name: str,
) -> List[Dict[str, Any]]:
    sample_id_response_dict = {
        response_unit.sample_id: response_unit.generated_text
        for response_unit in response_units
    }

    payloads: List[Dict[str, Any]] = []
    for datum in source_model.data:
        if datum.sample_id in sample_id_response_dict:
            payload = datum.model_dump()
            payload["initialization_step"]["arguments"]["database_path"] = os.path.join(
                cache_file, dataset_name + ".sqlite"
            )
            payload["output"]  # place pred_func_calls
            # output field is gold answer
            payloads.append(payload)
    return payloads
