import os

from api_integrated_llm.helpers.database_helper.tools.slot_filling_sql_tools import (
    data_join,
)
from api_integrated_llm.helpers.database_helper.tools.sequencing_tools import (
    create_getter,
)
from api_integrated_llm.helpers.database_helper.tools.sql_query_components import (
    safe_name_columns,
    make_safe,
    database_get_connection,
    database_get_table,
    database_close_connection,
)


def rewrite_table_alias_column(identifier: str, alias_to_name_dict: dict) -> str:
    rewritten = identifier.split(".")
    table_alias = rewritten[0]
    rewritten = (
        alias_to_name_dict[table_alias]["modified_table_name"]
        + "_"
        + make_safe(rewritten[1])
    )
    return table_alias, rewritten


def initialize_active_data(
    condition_sequence: list, alias_to_table_dict: dict, database_path: str
) -> dict:
    """
    Initializes active data based on the provided condition sequence, alias-to-table mapping, and database path.

    This function checks the validity of the database file at the specified path and raises an exception if the file is not found.
    After validating the database path, the function processes the condition sequence and alias-to-table dictionary
    to return a dictionary of active data.

    Args:
        condition_sequence (list): A list of conditions (joins) to be processed for initializing the data.
        alias_to_table_dict (dict): A dictionary mapping aliases to their respective tables.
        database_path (str): The file path to the database that will be used for the initialization.

    Raises:
        Exception: If the database file cannot be found at the specified location.

    Returns:
        dict: A dictionary representing the initialized active data based on the conditions and table mappings.
    """

    if not os.path.isfile(database_path):
        raise Exception(f"failed, can't find database at location ({database_path})")

    # load tables and prefix column names
    connection = database_get_connection(database_path)

    tables_dict = {}
    for alias, table_data in alias_to_table_dict.items():
        table = database_get_table(connection, table_data["original_table_name"])
        table = safe_name_columns(table)
        modified_table = table.add_prefix(table_data["modified_table_name"] + "_")
        tables_dict[alias] = modified_table

    database_close_connection(connection)

    if len(condition_sequence) == 0:
        if alias == "":
            return table.to_dict(orient="list")
        else:
            return modified_table.to_dict(orient="list")

    current_table = None
    for j in condition_sequence:
        left_table, left_col = rewrite_table_alias_column(j[0], alias_to_table_dict)
        if current_table is None:
            current_table = tables_dict[left_table]
        right_table, right_col = rewrite_table_alias_column(j[1], alias_to_table_dict)

        # The order of the terms in the join condition may be flipped
        # from the order of the tables in the sequence
        if (
            left_col in current_table.columns
            and right_col in tables_dict[right_table].columns
        ):
            pass
        elif (
            right_col in current_table.columns
            and left_col in tables_dict[right_table].columns
        ):
            left_col, right_col = right_col, left_col

        current_table = data_join(
            data_a=current_table,
            data_b=tables_dict[right_table],
            left_column_name=left_col,
            right_column_name=right_col,
            join_type="inner",
        )  # TODO: make this work with the type from data

    if current_table is None:
        current_table = list(tables_dict.values())[0]
    current_table = safe_name_columns(current_table)
    return current_table.to_dict(orient="list")


def create_getters_from_table(table_metadata: list[dict]) -> dict[callable]:
    # Initialize the list of API calls that will be translated to open API specs to build the dataset
    template_api_calls = {}

    for column in table_metadata:
        col_name = column["column_name"]
        col_desc = column["column_description"]
        getter = create_getter(col_name, col_desc)
        template_api_calls[col_name] = getter
    return template_api_calls
