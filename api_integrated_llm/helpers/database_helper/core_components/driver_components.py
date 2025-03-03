from collections import Counter
from decimal import Decimal
from typing import Callable
import json
import numpy as np
from pandas import DataFrame, Series

from api_integrated_llm.helpers.database_helper.tools.sql_query_components import (
    database_close_connection,
    database_get_connection,
    make_query_safe,
)
from api_integrated_llm.helpers.database_helper.tools.docstring_parsing_utils import (
    parse_google_style_docstring,
)
from api_integrated_llm.helpers.database_helper.tools.sequencing_tools import (
    create_getter,
)


def parse_fcn(api_name: str, api_body: callable) -> dict:
    parsed_docstring = parse_google_style_docstring(api_body.__doc__)
    fcn_metadata = format_api_spec(api_name, parsed_docstring)
    return fcn_metadata


def format_api_spec(name: str, parsed_docstring):
    # This is to match the input format needed for scoring.py
    fcn_metadata = {
        "description": parsed_docstring["description"],
        "name": name,
    }
    parameters = {"properties": {}, "required": [], "type": "object"}
    for param_name, param_data in parsed_docstring["parameters"].items():
        parameters["properties"][param_name] = param_data
        parameters["required"].append(param_name)
    fcn_metadata["parameters"] = parameters

    output_parameters = {
        "properties": {"output_0": parsed_docstring["output_parameter"]}
    }
    fcn_metadata["output_parameters"] = output_parameters

    return fcn_metadata


def update_dynamic_pool(payload: dict, api_pool: dict, api_names: set):
    payload_names = set(payload["tools"].keys())
    new_names = payload_names.difference(api_names.intersection(payload_names))
    api_names = api_names.union(payload_names)
    payload_pool = set()
    for api_name, api_body in payload["tools"].items():
        payload_pool.add(api_name)
        if api_name in new_names:
            api_pool[api_name] = parse_fcn(api_name, api_body)
    payload["tools"] = list(payload_pool)
    return payload, api_pool, api_names


def add_callable_to_api_sequence(
    sequence: list[dict], callable_dict: dict[str, callable], alias_to_table_dict: dict
) -> list[dict]:
    api_pool = {}
    # Handle adding getters if necessary
    for v in alias_to_table_dict.values():
        new_cols = v["modified_column_names"]
        descriptions = v["column_descriptions"]
        dtypes = v["column_dtypes"]
        for new_col, desc, dtype in zip(new_cols, descriptions, dtypes):
            getter = create_getter(new_col, desc, column_dtype=dtype)
            api_pool[getter.__name__] = getter

    api_pool = dict(api_pool, **callable_dict)
    for s in sequence:
        fname = s["name"]
        fcn = api_pool[fname]
        s["fcn"] = fcn

    return sequence


def create_structured_api_call(
    fcn: Callable, name: str, args: dict, output: str
) -> dict:
    payload = {"fcn": fcn, "name": name, "arguments": args, "label": output}
    return payload


def execute_single_api(api_name: str, api_args: dict, api_pool: dict[str, callable]):
    try:
        arg_fcn = api_pool[api_name]
    except:
        raise Exception(f"Fcn {api_name} not in API pool. ")
    output = arg_fcn(**api_args)
    return output


def execute_api_stack(apis: list[dict], api_pool: dict[str, callable]):
    output_dict = {}
    for api in apis:
        api_args_filled_in = {}
        for arg_key, arg_val in api["arguments"].items():
            if (
                isinstance(arg_val, str)
                and arg_val.startswith("$")
                and arg_val.endswith("$")
            ):
                arg_val = arg_val.lstrip("$").rstrip("$")
                api_args_filled_in[arg_key] = output_dict[
                    arg_val
                ]  # Throw an error if the necessary argument isn't found
            else:
                api_args_filled_in[arg_key] = arg_val
        output = execute_single_api(api["name"], api_args_filled_in, api_pool)
        # Update the output dict with the result of this call
        output_dict[api["label"]] = output

    return output_dict


def safe_cast(obj):
    """
    Recursively convert numpy data types to base Python types for JSON serialization.

    Args:
        obj: A list, tuple, list of tuples, string, or numeric type (int/float).

    Returns:
        A JSON serializable version of the input, with numpy types converted to native Python types.
    """

    # If the object is a numpy scalar (e.g., np.int64, np.float64, etc.), convert it to a native type
    if isinstance(obj, np.generic):
        return obj.item()

    elif isinstance(obj, Decimal):
        return float(obj)

    # If the object is a numpy array, recursively apply the conversion to its elements
    elif isinstance(obj, np.ndarray):
        return [safe_cast(item) for item in obj]

    # If the object is a list or a tuple, recursively convert elements
    elif isinstance(obj, (list, tuple)):
        return [safe_cast(item) for item in obj]

    # If the object is a string or numeric, just return it (strings and numbers are JSON serializable)
    else:
        return obj


def validate_sql_output(database_file: str, query: str):
    """
    Run the sql query, simplify the result, check that it's jsonify-able
    """
    # First execute the sql query against the cached database
    # Note that this will fail if executed against the original database
    # The cached database has had all of the column names made safe,
    # so we need to also do the same with the query.
    safe_query = make_query_safe(query)
    conn = database_get_connection(database_file)
    cursor = conn.cursor()
    cursor.execute(safe_query)
    query_results = cursor.fetchall()
    database_close_connection(conn)

    arr = np.array(query_results)
    query_results = arr.squeeze().transpose().tolist()

    # Check that json.dump won't fail later
    try:
        json.dumps(query_results)
    except:
        query_results = safe_cast(query_results)

    return query_results


def validate_api_output(required_api_calls: list, api_pool: dict[str, callable]):
    """
    Run the api sequence, simplify the result, check that it's jsonify-able
    """

    # First check the api calls
    for api in required_api_calls:
        try:
            json.dumps(api["arguments"])
        except:
            for k, v in api["arguments"].items():
                api["arguments"][k] = safe_cast(v)

    # Execute the required api calls and simplify
    api_result_dict = execute_api_stack(required_api_calls, api_pool)

    # All outputs that aren't in inputs to another call will be stitched together
    # into the final output in the order in which they appear
    api_input_args = []
    for call in required_api_calls:
        args = list(call["arguments"].values())
        args = [a.lstrip("$").rstrip("$") for a in args if isinstance(a, str)]
        api_input_args.extend(args)
    output_results_list = []
    for result_key in api_result_dict.keys():
        if result_key not in api_input_args:
            # This is an output key
            output_results = simplify_and_check_serialization(
                api_result_dict[result_key]
            )
            output_results_list.append(output_results)

    if len(output_results_list) == 1:
        output_results_list = output_results_list[0]
    return output_results_list


def simplify_and_check_serialization(api_results):
    if isinstance(api_results, list) and len(api_results) == 1:
        api_results = api_results[0]
    elif isinstance(api_results, DataFrame) or isinstance(api_results, Series):
        api_results = api_results.squeeze()
        if len(api_results.shape) == 1:
            api_results = tuple(api_results.tolist())
    elif isinstance(api_results, dict):
        api_results = list(api_results.values())
        squeezed_list = []
        for a in api_results:
            if isinstance(a, list) and len(a) == 1:
                squeezed_list.append(a[0])
            else:
                squeezed_list.append(a)
        if len(squeezed_list) > 1:
            api_results = squeezed_list
        else:
            api_results = squeezed_list[0]

    if isinstance(api_results, tuple):
        api_results = list(api_results)

    # Check that json.dump won't fail later
    try:
        json.dumps(api_results)
    except:
        api_results = safe_cast(api_results)
    return api_results


def check_equality_without_order(results_version_1, results_version_2) -> bool:
    correct_answer = False
    if isinstance(results_version_1, list) and isinstance(results_version_2, list):
        arr1 = np.array(results_version_1)
        arr2 = np.array(results_version_2)
        if arr1.shape == arr2.shape:  # Only the same if they are the same shape
            if len(arr1.shape) > 1:  # Compare 2D arrays (multiple select statements)
                correct_answer = np.array_equal(arr1, arr2)
            else:  # Compare lists without order
                # TODO: do this more efficiently
                query_no_order = Counter([str(r) for r in results_version_1])
                api_no_order = Counter([str(r) for r in results_version_2])
                if query_no_order == api_no_order:
                    correct_answer = True
    else:  # Compare scalar values
        correct_answer = bool(results_version_1 == results_version_2)
    return correct_answer


def validate_output(
    database_file: str,
    query: str,
    required_api_calls: list,
    api_pool: dict[str, callable],
):
    query_results = validate_sql_output(database_file=database_file, query=query)
    api_results = validate_api_output(required_api_calls, api_pool)
    correct_answer = check_equality_without_order(
        results_version_1=query_results, results_version_2=api_results
    )

    if not correct_answer:
        print()
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("QUERY = ", query)
        print(
            "QUERY RESULTS = ",
            query_results if type(query_results) != list else query_results[:20],
        )
        print(
            "API RESULTS = ",
            api_results if type(api_results) != list else api_results[:20],
        )
        print("ANSWERS MATCH = ", correct_answer)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print()
        print()
    return correct_answer, query_results, api_results
