import re
from typing import Literal, Union

import numpy as np
import pandas as pd


def group_data_by(data: dict, key_name: str) -> dict:
    """Group data by values of a specified key
    Args:
        data (dict): Table of data to be grouped
        key_name (str): name of key to group by
    Returns:
        dict: data with additional layer of keys given by unique values in the column given by `key_name`
    """
    assert (
        key_name in data.keys()
    ), f"Key: {key_name} not found, choose from {data.keys()}. "

    df = pd.DataFrame(data)
    grouped_keys = list(set(data[key_name]))
    grouped_data = {}
    for key_val in grouped_keys:
        selected = df[df[key_name] == key_val]
        grouped_data[key_val] = selected.to_dict(orient="list")
    return grouped_data


def retrieve_data(
    data: dict, key_name: Union[str, list[str]], distinct: bool = False, limit: int = -1
) -> dict:
    """Returns contents of a data column
    Args:
        data (dict): input data table in dictionary format
        key_name (Union[str, list[str]]): key name or list of key names to select
        distinct (bool): whether to return only distinct values
        limit (int): only return the first 'limit' elements
    Returns:
        dict: selected subset of input data
    """
    if isinstance(key_name, str):
        # TODO: add distinct for multiple columns
        if distinct:  # Only keep unique values, but preserve order
            data = {key_name: list(dict.fromkeys(data[key_name]))}
        key_name = [key_name]

    selected_data = {}
    for key in key_name:
        if limit >= 0:
            selected_data[key] = data[key][:limit]
        else:
            selected_data[key] = data[key]
    return selected_data


def aggregate_list(
    data: list, agg_operation: callable, distinct: bool = False, limit: int = -1
) -> Union[float, int]:
    if distinct:
        data = select_unique_values(data)

    if limit >= 0:
        data = data[:limit]

    result = agg_operation(data)
    return result


AGGREGATION_TYPES = Literal[
    "min", "max", "sum", "mean", "count", "std", "argmin", "argmax"
]


def aggregate_data(
    data: Union[dict, list],
    aggregation_type: AGGREGATION_TYPES,
    key_name: str = "",
    distinct: bool = False,
    limit: int = -1,
) -> Union[float, int, pd.Series]:
    """Perform an aggregation on input data.
    If the input data is list-like, returns the value of the aggregation over the list index.
    If the input data is tabular, returns a numerical value for the aggregation over a column.
    If the data is grouped tables, then replace the values in the specified key with their aggregation result

        Args:
            data (Union[dict, list]): data to be aggregated
            key_name (str): name of key to aggregate
            aggregation_type (typing.Literal['min', 'max', 'sum', 'mean', 'count', 'std', 'argmin', 'argmax']): the type of aggregation to perform, must be one of ['min', 'max', 'sum', 'mean', 'count', 'std', 'argmin', 'argmax']
            distinct (bool): whether to aggregate only distinct values
            limit (int): limit the aggregation to the first 'limit' elements
        Returns:
            dict: Result of aggregation.
    """

    # Check request validity
    assert aggregation_type in [
        "min",
        "max",
        "sum",
        "mean",
        "std",
        "count",
        "argmin",
        "argmax",
    ], f"{aggregation_type} not a valid aggregation. "

    agg_operation = agg_dict[aggregation_type]

    if isinstance(data, list):
        return aggregate_list(data, agg_operation, distinct=distinct, limit=limit)

    if key_name == "":
        assert (
            aggregation_type == "count"
        ), f"Can't aggregate '*' with aggregation type {aggregation_type}"
        k = list(data.keys())[0]
        return aggregate_list(data[k], agg_operation, distinct=distinct, limit=limit)

    elif key_name in data.keys():
        assert isinstance(data[key_name], list)
        assert len(data[key_name]) > 0, "Empty data can't be aggregated. "

        return aggregate_data(
            data[key_name],
            aggregation_type,
            key_name=key_name,
            distinct=distinct,
            limit=limit,
        )

    else:
        groups = list(data.keys())
        assert isinstance(data[groups[0]], dict)
        assert key_name in data[groups[0]].keys()
        for group in groups:
            vals = data[group][key_name]
            if distinct:
                vals = select_unique_values(vals)
            grouped_val = agg_operation(vals)
            lth = len(
                data[group][key_name]
            )  # lth needs to match original length, not distinct length
            data[group][key_name] = [grouped_val] * lth

        return data


OPERATION_TYPES = Literal["substring", "strfrtime"]


def transform_data(
    data: dict, key_name: str, operation_type: OPERATION_TYPES, operation_args: dict
) -> dict:
    """
    Transform the data assigned to a key in the input table (dict) using the specified operation and operation_args

        Args:
            data (dict): data to be transformed
            key_name (str): name of key to transform
            operation_type (typing.Literal['substring', 'strfrtime']): the type of operation to perform, must be one of ['substring', 'strfrtime']
            operation_args (dict): any arguments required by the operation_type
        Returns:
            dict: original data with the values corresponding to the specified key transformed
    """

    assert operation_type in [
        "substring",
        "strfrtime",
    ], f"Operation type {operation_type} not supported. "
    if operation_type == "substring":
        start_index = operation_args.get("start_index")
        end_index = operation_args.get("end_index")
        data = transform_data_to_substring(data, key_name, start_index, end_index)
    elif operation_type == "strfrtime":
        raise NotImplementedError("strfrtime broadcast operation not implemented. ")
    return data


def transform_data_to_substring(
    data: dict, key_name: str, start_index: int, end_index: int
) -> dict:
    """
    Transform list of string values by taking substrings

        Args:
            data (dict): table containing the data to be transformed
            key_name (str): name of string valued key to transform
            start_index (int): start of substring
            end_index (int): end of substring, must be >= start_index
        Returns:
            dict: original table (dict) with the specified key values transformed
    """
    data[key_name] = [val[start_index:end_index] for val in data[key_name]]
    return data


JOIN_TYPES = Literal["inner", "outer", "left", "right"]


def data_join(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    left_column_name: str,
    right_column_name: str,
    join_type: JOIN_TYPES = "inner",
) -> pd.DataFrame:
    """
    Joins two Pandas DataFrames on a specified column.

        Args:
            data_a: First DataFrame.
            data_b: Second DataFrame.
            left_column_name: The column name to join on from the first data. None indicates to use the data index.
            right_column_name: The column name to join on from the second data. None indicates to use the data index.
            join_type: Type of join: 'inner', 'outer', 'left', or 'right'. Default is 'inner'.

        Returns:
            A new DataFrame resulting from the join.
    """
    if left_column_name is not None and right_column_name is not None:
        joined_df = pd.merge(
            data_a,
            data_b,
            left_on=left_column_name,
            right_on=right_column_name,
            how=join_type,
        )
    elif left_column_name is None and right_column_name is not None:
        joined_df = pd.merge(
            data_a, data_b, left_index=True, right_on=right_column_name, how=join_type
        )
    elif left_column_name is not None and right_column_name is None:
        joined_df = pd.merge(
            data_a, data_b, left_on=left_column_name, right_index=True, how=join_type
        )
    else:
        joined_df = pd.merge(
            data_a, data_b, left_index=True, right_index=True, how=join_type
        )
    return joined_df


CONDITIONS = Literal[
    "equal_to",
    "not_equal_to",
    "greater_than",
    "less_than",
    "greater_than_equal_to",
    "less_than_equal_to",
    "contains",
    "like",
]


def filter_data(data: dict, key_name: str, value: str, condition: CONDITIONS) -> dict:
    """

    This function applies a filter on the given key of the input data
    based on the provided condition and value.
    It returns a new table (dict) with the rows that meet the condition.

    Args:
        data (dict): The input data to filter.
        key_name (str): The key on which the filter will be applied.
        value: The value to compare against in the filtering operation.
        condition (typing.Literal['equal_to', 'not_equal_to', 'greater_than', 'less_than', 'greater_than_equal_to', 'less_than_equal_to', 'contains', 'like']): The condition to apply for filtering. It must be one of the following:
            - 'equal_to': Filters rows where the column's value is equal to the given value.
            - 'not_equal_to': Filters rows where the column's value is not equal to the given value.
            - 'greater_than': Filters rows where the column's value is greater than the given value.
            - 'less_than': Filters rows where the column's value is less than the given value.
            - 'greater_than_equal_to': Filters rows where the column's value is greater than or equal to the given value.
            - 'less_than_equal_to': Filters rows where the column's value is less than or equal to the given value.
            - 'contains': Filters rows where the column's value contains the given value (applies to strings).
            - 'like': Filters rows where the column's value matches a regex pattern (applies to strings).

    Returns:
        dict: A new table (dict) containing the rows from the input data that meet the specified condition.

    Raises:
        ValueError: If the `condition` is not one of the recognized options.
        KeyError: If `key_name` does not exist in the `data`.
    """
    assert (
        key_name in data.keys()
    ), f"Key: {key_name} not found, choose from {data.keys()}. "
    # Deal with str to int comparisons
    if type(data[key_name][0]) == str and type(value) == int:
        value = str(value)

    df = pd.DataFrame(data)

    if condition == "equal_to":
        df = df[df[key_name] == value]
    elif condition == "not_equal_to":
        df = df[df[key_name] != value]
    elif condition == "greater_than":
        df = df[(df[key_name].notna()) & (df[key_name] > value)]
    elif condition == "less_than":
        df = df[(df[key_name].notna()) & (df[key_name] < value)]
    elif condition == "greater_than_equal_to":
        df = df[(df[key_name].notna()) & (df[key_name] >= value)]
    elif condition == "less_than_equal_to":
        df = df[(df[key_name].notna()) & (df[key_name] <= value)]
    elif condition == "contains":
        df = df[df[key_name].str.contains(value, case=False, na=False)]
    elif condition == "like":
        df = df[df[key_name].apply(lambda x: compare_like_pattern(x, value))]

    return df.to_dict(orient="list")


def sort_data(data: dict, key_name: str = "", ascending: bool = False) -> dict:
    """Sort data by the values associated with the chosen key='key_name'
    If the input data is list-like, returns the sorted list.
    If the input data is tabular, returns the table with rows sorted by the values in column 'key_name'.
    If the data is grouped tables, then sort the groups by the value in 'key_name'

        Args:
            data (dict): table in json format
            key_name (str): name of key to sort by
            ascending (bool): whether to sort by ascending order

        Returns:
            dict: data sorted by chosen key
    """

    # Handle strings and currency
    # if isinstance(data[key_name][0], str):
    #     try:
    #         data[key_name] = [float(v.strip().lstrip('$').replace(',','')) for v in data[key_name]]
    #     except:
    #         pass

    if isinstance(data, list):
        reverse = not ascending
        return sorted(data, reverse=reverse)

    if key_name in data.keys():
        assert isinstance(data[key_name], list)
        assert len(data[key_name]) > 0, "Empty data can't be sorted. "

        df = pd.DataFrame(data)
        sorted_data = df.sort_values(by=key_name, ascending=ascending, kind="mergesort")
        return sorted_data.to_dict(orient="list")

    # groups = list(data.keys())
    # assert isinstance(data[groups[0]], dict)
    # assert key_name in data[groups[0]].keys()
    # for group in groups:
    #     vals = data[group][key_name]
    #     # global agg_operation
    #     grouped_val = agg_operation(vals)
    #     lth = len(
    #         data[group][key_name]
    #     )  # lth needs to match original length, not distinct length
    #     data[group][key_name] = [grouped_val] * lth

    return data

    # df = pd.DataFrame(data)
    # sorted_data = df.sort_values(by=key_name, ascending=ascending, kind='mergesort')
    # return sorted_data.to_dict(orient='list')


def select_unique_values(unique_array: list) -> list:
    """Return only the distinct elements from the input list.

    Args:
        unique_array (list): A list of input data

    Returns:
        list: The distinct elements of the input list
    """

    return list(dict.fromkeys(unique_array))


def truncate(truncate_array, n: int) -> Union[dict, list]:
    """Return the first `n` elements of a list-like object.

    Args:
        truncate_array: A list-like object.
        n (int): The number of rows/elements to return.

    Returns:
        list: The first `n` elements of the list-like object.

    Raises:
        ValueError: If n is negative.
        TypeError: If the input is not a list-like object.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer.")

    return truncate_array[:n]


# def truncate(data: Union[dict, list], n: int) -> Union[dict, list]:
#     """Return the first `n` rows of a table or the first `n` elements of a list-like object.

#     Args:
#         data (Union[dict, list): A table (dict) or a list-like object.
#         n (int): The number of rows/elements to return.

#     Returns:
#         Union[dict, list]: The first `n` rows of the table (dict)
#                                           or the first `n` elements of the list-like object.

#     Raises:
#         ValueError: If n is negative.
#         TypeError: If the input is neither a dict nor a list-like object.
#     """
#     if n < 0:
#         raise ValueError("n must be a non-negative integer.")

#     if isinstance(data, dict):
#         return pd.DataFrame(data).head(n).to_dict(orient='list')
#     elif isinstance(data, list):
#         return data[:n]
#     else:
#         raise TypeError("Input must be a dict or a list-like object.")


def compare_like_pattern(pattern: str, comparison_value: str) -> bool:
    # Escape special regex characters in the pattern, except for % and _
    pattern = re.escape(pattern)

    # Replace SQL wildcards with equivalent regex patterns
    pattern = pattern.replace(r"\%", ".*")  # % becomes .*
    pattern = pattern.replace(r"\_", ".")  # _ becomes .

    # Add start and end anchors to match the whole string
    pattern = "^" + pattern + "$"

    # Compile and use the regex to match the string
    return bool(re.match(pattern, comparison_value))


##############################################################################################################


# Aggregation related functions


##############################################################################################################


def get_min(data: list[float]):
    """Return the minimum value from a list.

    Args:
        data (list): A list of numbers.

    Returns:
        float: The minimum value in the input data.
    """
    return np.nanmin(data)


def get_max(data: list[float]):
    """Return the maximum value from a list.

    Args:
        data (list): A list of numbers.

    Returns:
        float: The maximum value in the input data.
    """
    return np.nanmax(data)


def get_count(data: list[float]):
    """Return the number of elements in a list.

    Args:
        data (list): A list of numbers.

    Returns:
        int: The number of elements in the input data.
    """
    return len(data)


def get_sum(data: list[float]):
    """Return the sum of values in a list.

    Args:
        data (list): A list of numbers.

    Returns:
        float: The sum of the input data.
    """
    return np.nansum(data)


def get_mean(data: list[float]):
    """Return the mean of values in a list.

    Args:
        data (list): A list of numbers.

    Returns:
        float: The mean of the input data.
    """
    return np.nanmean(data)


def get_std(data: list[float]):
    """Return the standard deviation of values in a list.

    Args:
        data (list): A list of numbers.

    Returns:
        float: The standard deviation of the input data.
    """
    return np.nanstd(data)


def get_argmin(data: list[float]) -> int:
    """Return the index of the minimum value in a list.

    Args:
        data (list): A list of numbers.

    Returns:
        int: The index of the minimum value in the input data.
    """
    return list(data).index(min(data))


def get_argmax(data: list[float]) -> int:
    """Return the index of the maximum value in a list.

    Args:
        data (list): A list of numbers.

    Returns:
        int: The index of the maximum value in the input data.
    """
    # Cast to list makes this work if data is a 1-D numpy array.
    return list(data).index(max(data))


agg_dict = {
    "min": get_min,
    "max": get_max,
    "count": get_count,
    "sum": get_sum,
    "mean": get_mean,
    "std": get_std,
    "argmin": get_argmin,
    "argmax": get_argmax,
}
