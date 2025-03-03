import logging
import os
from typing import Callable

import sqlglot
import sqlglot.expressions

from api_integrated_llm.helpers.database_helper.tools.slot_filling_sql_tools import (
    aggregate_data,
    filter_data,
    sort_data,
    retrieve_data,
    transform_data,
    select_unique_values,
    group_data_by,
)
from api_integrated_llm.helpers.database_helper.tools.sql_tools import (
    initialize_active_data,
)

from api_integrated_llm.helpers.database_helper.database_builders.sql_dataset_builder import (
    SqlDatasetBuilder,
    identify_aggregation_expression,
)
from api_integrated_llm.helpers.database_helper.core_components.driver_components import (
    create_structured_api_call,
)


class SqlSlotFillingDatasetBuilder(SqlDatasetBuilder):
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
        base_fcns = {
            f.__name__: f
            for f in [
                aggregate_data,
                filter_data,
                sort_data,
                retrieve_data,
                select_unique_values,
                transform_data,
                group_data_by,
                initialize_active_data,
            ]
        }
        wrapped_fcns = self.add_read_from_file_decorator(base_fcns)
        self.available_function_dict = wrapped_fcns

    def build(self) -> None:
        # Initialize the list of API calls that will be translated to open API specs to build the dataset

        # column_metadata = []
        # for table_name, table_metadata in self.loader.table_descriptions.items():
        #     for m in table_metadata:
        #         col_name = m['column_name']
        #         modified_col_name = table_name + "_" + col_name
        #         col_desc = m['column_description']
        #         column_metadata.append({"name": modified_col_name, "description": col_desc})

        # def get_available_keys_and_descriptions_template(data_descriptions: list[dict]) -> list[dict]:
        #     """Lookup function to get keys and descriptions of available data (table columns)

        #         Returns:
        #             list[dict]: A list of dictionaries, one per column in the data table, containing the 'name' and 'description' for that column.
        #     """
        #     return data_descriptions

        # fcn_keys_and_descriptions = partial(get_available_keys_and_descriptions_template, data_descriptions=column_metadata)
        # update_wrapper(fcn_keys_and_descriptions, get_available_keys_and_descriptions_template)
        # fcn_keys_and_descriptions.__name__ = "retrieve_available_keys_and_descriptions"
        # self.available_function_dict["retrieve_available_keys_and_descriptions"] = fcn_keys_and_descriptions

        # def get_available_keys_template(data_keys: list[str]) -> list[str]:
        #     """Lookup function to get keys available data (table columns)

        #         Returns:
        #             list[str]: A list of column keys available in the data table
        #     """
        #     return data_keys

        # fcn_keys = partial(get_available_keys_template, data_keys=[c['name'] for c in column_metadata])
        # update_wrapper(fcn_keys, get_available_keys_template)
        # fcn_keys.__name__ = "retrieve_available_keys"
        # self.available_function_dict["retrieve_available_keys"] = fcn_keys

        # def get_description_by_key_template(data_dict: dict[str, str], key: str) -> str:
        #     """Lookup function to get the description of the data assigned to column 'key'

        #         Args:
        #             key (str): the data key (column name)
        #         Returns:
        #             str: The description of the data in column 'key'
        #     """
        #     descriptions = data_dict
        #     description = descriptions.get(key, f"{key} not available. ")
        #     return description

        # data_dictionary = {c['name']: c['description'] for c in column_metadata}
        # fcn_description = partial(get_description_by_key_template, data_dict=data_dictionary)
        # update_wrapper(fcn_description, get_description_by_key_template)
        # fcn_description.__name__ = "retrieve_description_by_key"
        # self.available_function_dict["retrieve_description_by_key"] = fcn_description

        # Dropping these functions from the pool.
        pass

    def set_query_specific_api_pool(
        self, api_sequence: list[dict]
    ) -> dict[str, callable]:
        key_names_and_descriptions = self.set_query_specific_columns_and_descriptions(
            api_sequence
        )

        # For slot-filling version this is the same for every data point.
        return self.available_function_dict, key_names_and_descriptions

    def translate_query_from_sql_tree(self, query: str) -> list:
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
        required_api_calls = []
        db_path = os.path.join(
            self.database_cache_location, self.database_name + ".sqlite"
        )
        required_api_calls.append(
            create_structured_api_call(
                initialize_active_data,
                initialize_active_data.__name__,
                {
                    "condition_sequence": join_sequence,
                    "alias_to_table_dict": simplified_dict,
                    "database_path": db_path,
                },
                STARTING_TABLE_VAR,
            )
        )

        # Handle 'where'
        # where_calls = self._process_where_clause(glot_ast, required_api_calls[-1]['label'])
        where_calls = self.process_where_clauses(
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
        get_agg_fcn = self._process_select_and_aggregate(
            parsed_select, required_api_calls[-1]["label"]
        )
        required_api_calls.extend(get_agg_fcn)

        return required_api_calls

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

    def _process_select_and_aggregate(
        self, parsed_select: dict, table_var: str
    ) -> dict:
        api_calls = []

        limit = parsed_select["limit"]
        distinct = parsed_select["distinct"]

        for idx, clause in enumerate(parsed_select["clauses"]):
            # Handle SUBSTR conditions
            #     if isinstance(eq.this, sqlglot.expressions.Substring):
            #         column_str = str(eq.this.this)
            #         comparison_column = self._reformat_compound_column_name(column_str)
            #         start = eq.this.args['start'].to_py() - 1  # SQL values will be 1-indexed, convert to 0-index
            #         length = eq.this.args['length'].to_py()
            #         operation_args = {"start_index": start, "end_index": start+length}
            #         substring_args = {"data_source": f'${input_df_key}$', "key_name": comparison_column, "operation_type": "substring", "operation_args": operation_args}
            #         api = create_structured_api_call(transform_data, transform_data.__name__, substring_args, 'PROCESSED_DF')
            #         api_calls.append(api)
            #         table_name = '$PROCESSED_DF$'
            col_name = clause[0]
            agg_expr = clause[1]
            select_args = {
                "data_source": f"${table_var}$",
                "key_name": col_name,
                "distinct": distinct,
                "limit": limit,
            }
            if agg_expr:
                select_args["aggregation_type"] = agg_expr
                get_agg_fcn = create_structured_api_call(
                    aggregate_data,
                    aggregate_data.__name__,
                    select_args,
                    agg_expr.upper(),
                )
                api_calls.append(get_agg_fcn)
            else:
                api_calls.append(
                    create_structured_api_call(
                        retrieve_data,
                        retrieve_data.__name__,
                        select_args,
                        "SELECT_COL_" + str(idx),
                    )
                )

        return api_calls

    def process_where_clauses(
        self, ast: sqlglot.Expression, input_df_key: str
    ) -> list[Callable]:
        if "where" not in ast.args:
            return []

        table_name = f"${input_df_key}$"
        where_apis = []
        where = ast.args["where"]
        parsed_where = self.parse_where_clause(where)
        for i, clause in enumerate(parsed_where["clauses"]):
            comparison_column = clause[0]
            value = clause[1]
            comparison_operator = clause[2]
            filter_args = {
                "data_source": table_name,
                "key_name": comparison_column,
                "value": value,
                "condition": comparison_operator,
            }
            output_df = "FILTERED_DF_" + str(i)
            api = create_structured_api_call(
                filter_data, filter_data.__name__, filter_args, output_df
            )
            table_name = "$" + output_df + "$"
            where_apis.append(api)

        return where_apis

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
        ascending = not bool(orderby.args["desc"])  # return results in descending order
        agg = identify_aggregation_expression(orderby.args["this"])
        if agg:  # We need an aggregation before we orderby
            orderby_column = self.process_column_object(orderby.args["this"].this)
            get_agg_args = {
                "data_source": f"${input_df_key}$",
                "key_name": orderby_column,
                "aggregation_type": agg,
                "distinct": False,
                "limit": -1,
            }
            get_agg_fcn = create_structured_api_call(
                aggregate_data, aggregate_data.__name__, get_agg_args, agg.upper()
            )
            api_calls.append(get_agg_fcn)
            table_name = f"${agg.upper()}$"
        else:
            orderby_column = self.process_column_object(orderby.this)
            table_name = f"${input_df_key}$"

        orderby_args = {
            "data_source": table_name,
            "key_name": orderby_column,
            "ascending": ascending,
        }
        api = create_structured_api_call(
            sort_data, sort_data.__name__, orderby_args, "SORTED_DF"
        )
        api_calls.append(api)
        return api_calls
