from functools import partial, update_wrapper
from typing import Callable, get_args

from api_integrated_llm.helpers.database_helper.tools.slot_filling_sql_tools import (
    sort_data,
    filter_data,
    aggregate_data,
    CONDITIONS,
    AGGREGATION_TYPES,
    agg_dict,
)


def fill_slots_filter_data() -> dict[str, Callable]:
    filled_in_fcns = {}
    for f in get_args(CONDITIONS):
        name = "select_data_" + f
        filter_fcn = partial(filter_data, condition=f)
        update_wrapper(filter_fcn, filter_data)
        filter_fcn.__name__ = name
        # Parse the docstring
        doc = filter_fcn.__doc__
        a = doc.split("condition (typing.Literal")
        prefix = a[0]
        b = a[1].split("Returns:")
        middle = b[0].split("one of the following:")[1].split("-")
        middle = [m.strip() for m in middle]
        our_condition = [p for p in middle if p.startswith(f"'{f}'")]
        assert len(our_condition) == 1
        desc = our_condition[0].split(": ")[1]
        newdoc = (
            desc + "\n     Args:" + prefix.split("Args:")[1] + "\n   Returns:" + b[1]
        )
        filter_fcn.__doc__ = newdoc
        filter_fcn.__annotations__["condition"] = f
        filled_in_fcns[name] = filter_fcn
    return filled_in_fcns


def fill_slots_sort_data() -> dict[str, Callable]:
    filled_in_fcns = {}
    for direction in ["ascending", "descending"]:
        name = "sort_data_" + direction
        ascending = direction == "ascending"
        sort_fcn = partial(sort_data, ascending=ascending)
        update_wrapper(sort_fcn, sort_data)
        sort_fcn.__name__ = name
        filled_in_fcns[name] = sort_fcn
    return filled_in_fcns


def fill_slots_aggregate_data() -> dict[str, Callable]:
    filled_in_fcns = {}
    for agg in get_args(AGGREGATION_TYPES):
        name = "compute_data_" + agg
        agg_fcn = partial(aggregate_data, aggregation_type=agg, limit=-1)
        update_wrapper(agg_fcn, aggregate_data)
        agg_fcn.__name__ = name
        agg_fcn.__doc__ = agg_dict[agg].__doc__
        filled_in_fcns[name] = agg_fcn
    return filled_in_fcns


def create_getter(
    column: str, column_description: str, column_dtype: str = None
) -> Callable:
    def get_column(data: dict, key_name: str) -> list:
        return data[key_name]

    getter_func = partial(get_column, key_name=column)
    getter_name = f"get_{column}s"
    if column_dtype is None:
        getter_return_type = list
    else:
        getter_return_type = list[column_dtype]
    getter_docstring = (
        f"Lookup data {column}: {column_description}"
        + "\nReturns:\n\t"
        + f"{getter_return_type}"
    )
    getter_func.__name__ = getter_name
    getter_func.__doc__ = getter_docstring
    return getter_func
