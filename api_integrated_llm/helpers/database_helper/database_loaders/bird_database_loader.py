from datetime import datetime
from numpy import isnan
import json
import os
import pandas as pd
from typing import Any

from api_integrated_llm.helpers.database_helper.database_loaders.database_loader import (
    DatabaseLoader,
)
from api_integrated_llm.helpers.database_helper.tools.sql_query_components import (
    make_safe,
)

dtype_translator = {
    "text": str,
    "integer": int,
    "real": float,
    "date": datetime,
    "datetime": datetime,
    "": Any,
    "nan": Any,
}


class BirdDatabaseLoader(DatabaseLoader):
    def __init__(
        self,
        database_name: str,
        database_location: str,
        database_cache_location: str = ".",
    ):
        super().__init__(
            database_name,
            database_location,
            database_cache_location=database_cache_location,
        )

        # Verify the database exists
        self.database_path = os.path.join(
            self.database_location, self.name, self.name + ".sqlite"
        )
        assert os.path.isfile(
            self.database_path
        ), f"Database: {self.database_path} was not found. "

    def _load_keys(self):
        # Load keys
        if os.path.isfile(os.path.join(self.database_location, "train_tables.json")):
            keys_file = os.path.join(self.database_location, "train_tables.json")
        elif os.path.isfile(os.path.join(self.database_location, "dev_tables.json")):
            keys_file = os.path.join(self.database_location, "dev_tables.json")
        assert os.path.isfile(keys_file), f"{keys_file} not found. "
        with open(keys_file) as f:
            key_data = json.load(f)

        # Grab the section for the database we are loading
        for db in key_data:
            if db["db_id"] == self.name:
                key_data = db
                break

        tables = db["table_names_original"]
        columns = pd.DataFrame(
            db["column_names_original"], columns=["table_id", "column_name"]
        )
        columns["column_name"] = columns["column_name"].apply(make_safe)
        self.column_list = list(
            set(columns["column_name"].to_list())
        )  # Only unique column names

        # Primary and foreign keys are encoded by their index in the total list of database columns
        # Need to translate this global index into a table-column name pair.
        primary_keys = db["primary_keys"]
        foreign_keys = db["foreign_keys"]

        # Note that a primary key may be composed of multiple columns
        for p in primary_keys:
            if not isinstance(p, list):
                p = [p]
            tab = None
            primary_key_columns = []
            for ind in p:  # Loop over columns in primary key, usually will only be 1
                col = columns.iloc[ind]
                if (
                    tab is not None
                ):  # Multi-column key, ensure we're still in the same table
                    assert tab == tables[col["table_id"]]
                tab = tables[col["table_id"]]
                col_name = col["column_name"]
                primary_key_columns.append(col_name)
            self.primary_keys[tab].append(primary_key_columns)

        for f in foreign_keys:
            col1 = columns.iloc[f[0]]["column_name"]
            tab1 = tables[columns.iloc[f[0]]["table_id"]]

            col2 = columns.iloc[f[1]]["column_name"]
            tab2 = tables[columns.iloc[f[1]]["table_id"]]
            self.foreign_keys.append(
                [{"table": tab1, "column": col1}, {"table": tab2, "column": col2}]
            )

    def load_lazy(self):
        # Load individual table descriptions
        table_description_directory = os.path.join(
            self.database_location, self.name, "database_description"
        )
        assert os.path.isdir(
            table_description_directory
        ), f"{table_description_directory} is not a directory. "
        files = os.listdir(table_description_directory)
        files = [f for f in files if not f.startswith(".")]
        files = [
            f
            for f in files
            if os.path.isfile(os.path.join(table_description_directory, f))
        ]

        for f in files:
            table_metadata = []
            table_name = f.replace(".csv", "")
            try:
                description_df = pd.read_csv(
                    os.path.join(table_description_directory, f)
                )
            except UnicodeDecodeError:
                try:
                    description_df = pd.read_csv(
                        os.path.join(table_description_directory, f), encoding="latin1"
                    )
                except:
                    raise Exception(
                        "Error in pandas read_csv. Try a different encoding scheme"
                    )
            for col in description_df.to_dict(orient="records"):
                format = col["data_format"]
                if isinstance(format, str):
                    format = format.strip()
                dtype = dtype_translator.get(format, None)
                if dtype is None:
                    try:
                        if isnan(format):
                            dtype = float
                    except:
                        dtype = object
                safe_name = make_safe(col["original_column_name"])
                metadata = {
                    "column_name": safe_name,
                    "column_description": col["column_description"],
                    "column_dtype": dtype,
                }
                table_metadata.append(metadata)
            self.table_descriptions[table_name] = table_metadata

        self._load_keys()
