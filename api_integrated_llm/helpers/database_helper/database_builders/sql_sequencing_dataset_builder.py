import logging
import os
from typing import Callable

import sqlglot

from api_integrated_llm.helpers.database_helper.tools.slot_filling_sql_tools import (
    transform_data_to_substring,
    truncate,
    select_unique_values,
    group_data_by,
)
from api_integrated_llm.helpers.database_helper.tools.sql_tools import (
    initialize_active_data,
)
from api_integrated_llm.helpers.database_helper.tools.sequencing_tools import (
    fill_slots_filter_data,
    fill_slots_sort_data,
    fill_slots_aggregate_data,
    create_getter,
)

from api_integrated_llm.helpers.database_helper.database_builders.sql_dataset_builder import (
    SqlDatasetBuilder,
    identify_aggregation_expression,
)
from api_integrated_llm.helpers.database_helper.core_components.driver_components import (
    create_structured_api_call,
)


class SqlSequencingDatasetBuilder(SqlDatasetBuilder):
    def __init__(
        self,
        database_name: str,
        dataset_path: str,
        cache_location: str = None,
        source_dataset_name: str = None,
    ) -> None:
        super().__init__(
            database_name,
            dataset_path,
            cache_location=cache_location,
            source_dataset_name=source_dataset_name,
        )
        self.logger = logging.getLogger(__name__)

        # These functions can stay as is, they are already applied to generic (list-like) objects.
        self.available_function_dict = self.add_read_from_file_decorator(
            {
                f.__name__: f
                for f in [
                    group_data_by,
                    transform_data_to_substring,
                    initialize_active_data,
                ]
            }
        )
        self.available_function_dict[truncate.__name__] = truncate
        self.available_function_dict[
            select_unique_values.__name__
        ] = select_unique_values

    def build(self) -> None:
        # Bind function args to create api pool
        bound_filter_fcns = self.add_read_from_file_decorator(fill_slots_filter_data())
        bound_sort_fcns = self.add_read_from_file_decorator(fill_slots_sort_data())
        bound_agg_fcns = self.add_read_from_file_decorator(fill_slots_aggregate_data())
        self.available_function_dict.update(bound_filter_fcns)
        self.available_function_dict.update(bound_sort_fcns)
        self.available_function_dict.update(bound_agg_fcns)

    def set_query_specific_api_pool(
        self, api_sequence: list[dict]
    ) -> dict[str, callable]:
        key_names_and_descriptions = self.set_query_specific_columns_and_descriptions(
            api_sequence
        )
        # Initialize the rest of the API pool
        template_api_calls = {}
        for row in key_names_and_descriptions:
            getter = create_getter(row["key_name"], row["description"], row["dtype"])
            template_api_calls[getter.__name__] = getter
        template_api_calls = self.add_read_from_file_decorator(template_api_calls)

        available_api_dict = dict(template_api_calls, **self.available_function_dict)
        return available_api_dict, key_names_and_descriptions

    def translate_query_from_sql_tree(self, query: str) -> dict:
        if query.count("SELECT") > 1:
            raise Exception(
                "Can't support multiple SELECT statements in single query. "
            )

        (
            glot_ast,
            query,
            alias_to_table_dict,
            join_sequence,
        ) = self.parse_query_and_get_join_metadata(query)
        simplified_dict = self.simplify_alias_to_table_dict(alias_to_table_dict)
        STARTING_TABLE_VAR = "starting_table_var"

        # Load the starting table
        db_path = os.path.join(
            self.database_cache_location, self.database_name + ".sqlite"
        )
        required_api_calls = []
        required_api_calls.append(
            create_structured_api_call(
                self.available_function_dict["initialize_active_data"],
                initialize_active_data.__name__,
                {
                    "condition_sequence": join_sequence,
                    "alias_to_table_dict": simplified_dict,
                    "database_path": db_path,
                },
                STARTING_TABLE_VAR,
            )
        )

        # Add the getters
        available_api_dict, _ = self.set_query_specific_api_pool(required_api_calls)
        # Handle 'where'
        where_calls = self._process_where_clause(
            glot_ast, required_api_calls[-1]["label"]
        )
        required_api_calls.extend(where_calls)

        # Handle 'group by'
        groupby_calls = self._process_groupby(glot_ast, required_api_calls[-1]["label"])
        required_api_calls.extend(groupby_calls)

        # Handle 'order'
        order_calls = self._process_orderby_clause(
            glot_ast, required_api_calls[-1]["label"]
        )
        required_api_calls.extend(order_calls)

        # Create a get for the select
        parsed_select = self.parse_select_clause(glot_ast)
        select_calls = self._process_select_and_aggregate(
            parsed_select, required_api_calls[-1]["label"], available_api_dict
        )
        required_api_calls.extend(select_calls)

        return required_api_calls

    def _process_select_and_aggregate(
        self,
        parsed_select: dict,
        table_var: str,
        available_api_dict: dict[str, callable],
    ) -> dict:
        api_calls = []

        limit = parsed_select["limit"]
        distinct = parsed_select["distinct"]  # TODO: handle distinct

        for idx, clause in enumerate(parsed_select["clauses"]):
            col_name = clause[0]
            agg_expr = clause[1]
            if agg_expr:
                agg_fcn = available_api_dict["compute_data_" + agg_expr]
                agg_fcn = create_structured_api_call(
                    agg_fcn,
                    agg_fcn.__name__,
                    {
                        "data_source": f"${table_var}$",
                        "key_name": col_name,
                        "distinct": distinct,
                    },
                    agg_expr.upper(),
                )
                api_calls.append(agg_fcn)
            else:
                getter_name = f"get_{col_name}s"
                getter_fcn = available_api_dict[getter_name]
                get_fcn_struct = create_structured_api_call(
                    getter_fcn,
                    getter_fcn.__name__,
                    {"data_source": f"${table_var}$"},
                    "SELECT_COL_" + str(idx),
                )
                api_calls.append(get_fcn_struct)
                if distinct:
                    distinct_fcn = create_structured_api_call(
                        self.available_function_dict["select_unique_values"],
                        select_unique_values.__name__,
                        {"unique_array": f"${api_calls[-1]['label']}$"},
                        "DISTINCT_COL_" + str(idx),
                    )
                    api_calls.append(distinct_fcn)

        if limit != -1:
            input_df_key = api_calls[-1]["label"]
            limit_fcn = create_structured_api_call(
                truncate,
                truncate.__name__,
                {"truncate_array": f"${input_df_key}$", "n": limit},
                "LIMIT",
            )
            api_calls.append(limit_fcn)

        return api_calls

    def _process_groupby(self, ast: sqlglot.Expression, table_var: str) -> dict:
        if "group" not in ast.args:
            return []

        expression = ast.args["group"].expressions[0]
        get_groupby_column = self.process_column_object(expression)
        groupby_args = {"data_source": f"${table_var}$", "key_name": get_groupby_column}
        groupby_fcn = create_structured_api_call(
            group_data_by, group_data_by.__name__, groupby_args, "GROUPED"
        )
        return [groupby_fcn]

    def _process_single_filter(
        self, eq: sqlglot.expressions.EQ, input_df_key: str
    ) -> list[Callable]:
        api_calls = []
        try:
            value = eq.expression.to_py()
        except:
            value = eq.expression.this.this
        # Handle SUBSTR conditions
        if isinstance(eq.this, sqlglot.expressions.Substring):
            column_str = str(eq.this.this)
            comparison_column = self._reformat_compound_column_name(column_str)
            start = (
                eq.this.args["start"].to_py() - 1
            )  # SQL values will be 1-indexed, convert to 0-index
            length = eq.this.args["length"].to_py()
            substring_args = {
                "data_source": f"${input_df_key}$",
                "key_name": comparison_column,
                "start_index": start,
                "end_index": start + length,
            }
            api = create_structured_api_call(
                self.available_function_dict["transform_data_to_substring"],
                transform_data_to_substring.__name__,
                substring_args,
                "PROCESSED_DF",
            )
            api_calls.append(api)
            table_name = "$PROCESSED_DF$"
        else:
            column_str = str(eq.this)
            comparison_column = self._reformat_compound_column_name(column_str)
            table_name = f"${input_df_key}$"

        expression = str(eq)
        # Extract the appropriate comparison operator from the sql expression
        if (">=") in expression:
            filter_op = self.available_function_dict[
                "select_data_greater_than_equal_to"
            ]
        elif ("<=") in expression:
            filter_op = self.available_function_dict["select_data_less_than_equal_to"]
        elif ("<>") in expression:
            filter_op = self.available_function_dict["select_data_not_equal_to"]
        elif (">") in expression:
            filter_op = self.available_function_dict["select_data_greater_than"]
        elif ("<") in expression:
            filter_op = self.available_function_dict["select_data_less_than"]
        elif ("BETWEEN") in expression:
            raise Exception("Haven't implemented BETWEEN filter")
        elif ("LIKE") in expression:
            filter_op = self.available_function_dict["select_data_like"]
        else:
            assert ("=" in expression) or (
                "IS" in expression
            ), f"Couldn't find = in {expression}"
            filter_op = self.available_function_dict["select_data_equal_to"]

        filter_args = {
            "data_source": table_name,
            "key_name": comparison_column,
            "value": value,
        }
        api = create_structured_api_call(
            filter_op, filter_op.__name__, filter_args, "FILTERED_DF"
        )
        api_calls.append(api)
        return api_calls

    def _process_orderby_clause(
        self, ast: sqlglot.Expression, input_df_key: str
    ) -> list[Callable]:
        if "order" not in ast.args:
            return []

        api_calls = []
        orderby = ast.args["order"].expressions[0]
        assert (
            len(ast.args["order"].expressions) == 1
        ), "Need to implement multiple 'order by' clauses"
        if bool(orderby.args["desc"]):
            sort_operator = self.available_function_dict["sort_data_descending"]
        else:
            sort_operator = self.available_function_dict["sort_data_ascending"]

        agg = identify_aggregation_expression(orderby.args["this"])
        if agg is not None:
            orderby_column = self.process_column_object(orderby.args["this"].this)
            get_agg_args = {
                "data_source": f"${input_df_key}$",
                "key_name": orderby_column,
            }
            agg_fcn = self.available_function_dict["compute_data_" + agg]
            agg_api_call = create_structured_api_call(
                agg_fcn, agg_fcn.__name__, get_agg_args, agg.upper()
            )
            api_calls.append(agg_api_call)
            table_name = f"${agg.upper()}$"
        else:
            orderby_column = self.process_column_object(orderby.this)
            table_name = f"${input_df_key}$"

        orderby_args = {"data_source": table_name, "key_name": orderby_column}
        api = create_structured_api_call(
            sort_operator, sort_operator.__name__, orderby_args, "SORTED_DF"
        )
        api_calls.append(api)
        return api_calls
