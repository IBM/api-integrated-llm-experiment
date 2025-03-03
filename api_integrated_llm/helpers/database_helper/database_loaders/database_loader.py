from collections import defaultdict
from datetime import datetime
import os
import pandas as pd

from api_integrated_llm.helpers.database_helper.tools.sql_query_components import (
    database_get_connection,
    database_close_connection,
    database_lookup_tables,
    database_get_table,
    safe_name_columns,
)

dtype_translator_sparc = {
    "text": str,
    "integer": int,
    "number": float,
    "date": datetime,
    "datetime": datetime,
    "time": datetime,
    "boolean": bool,
    "": object,
    "nan": object,
    "other": object,
}


class DatabaseLoader:
    def __init__(
        self,
        database_name: str,
        database_location: str,
        database_cache_location: str = ".",
    ):
        assert os.path.isdir(
            database_location
        ), f"{database_location} is not a directory. "
        self.database_location = database_location
        self.database_cache_location = database_cache_location
        self.cache_file = os.path.join(
            self.database_cache_location, database_name + ".sqlite"
        )
        self.name = database_name
        self.column_list = []
        self.table_descriptions = {}
        self.table_data = {}
        self.primary_keys = defaultdict(list)
        self.foreign_keys = []

    def get_table_column_descriptions(self):
        table_column_descriptions = {}
        for table, cols in self.table_descriptions.items():
            table_column_descriptions[table] = {}
            for row in cols:
                table_column_descriptions[table][row["column_name"]] = {
                    "column_description": row["column_description"],
                    "column_dtype": row["column_dtype"],
                }
        return table_column_descriptions

    def _load_keys(self, key_data: dict):
        raise NotImplementedError()

    def load(self, load_in_memory: bool = False):
        self.load_lazy()
        if load_in_memory:
            # Convert tables in database into pandas dataframes
            for table_name in self.table_descriptions.keys():
                table_data = safe_name_columns(self.get_table_as_dataframe(table_name))
                self.table_data[table_name] = table_data

        elif self.database_cache_location is not None:
            assert os.path.isdir(
                self.database_cache_location
            ), f"database_cache_location = {self.database_cache_location} is not a valid directory. "

            connection = database_get_connection(self.cache_file)
            for table_name in self.table_descriptions.keys():
                loaded_table = self.get_table_as_dataframe(
                    table_name, use_original_not_cache=True
                )
                safe_table = safe_name_columns(loaded_table)
                safe_table.to_sql(
                    table_name, connection, if_exists="replace", index=False
                )
            database_tables = database_lookup_tables(connection)
            assert set(database_tables) == set(
                self.table_descriptions.keys()
            ), f"Database tables: {set(database_tables)} not equal to self.table_descriptions: {set(self.table_descriptions.keys())}"
            database_close_connection(connection)

    def load_lazy(self):
        raise NotImplementedError()

    def get_table_as_dataframe(
        self, table_name: str, use_original_not_cache: bool = False, safe: bool = True
    ) -> pd.DataFrame:
        if use_original_not_cache:
            assert (
                self.database_path is not None
            ), "Can't call get_table without calling lazy_loading first"
            path_to_database = self.database_path
        else:
            path_to_database = self.cache_file

        connection = database_get_connection(path_to_database)
        table = database_get_table(connection, table_name)

        # Close the database
        database_close_connection(connection)

        if safe:
            table = safe_name_columns(table)
        return table

    def get_table_as_dict(
        self, table_name: str, use_original_not_cache: bool = False, safe: bool = True
    ) -> dict:
        table = self.get_table_as_dataframe(
            table_name, use_original_not_cache=use_original_not_cache, safe=safe
        )
        table = table.to_dict(orient="list")
        return table

    def add_table_to_cache(
        self, table: pd.DataFrame, table_name: str, column_info: list[dict] = None
    ):
        table = safe_name_columns(table)
        conn = database_get_connection(self.cache_file)
        table.to_sql(table_name, conn, if_exists="replace", index=False)
        conn.close()
        self.table_descriptions[table_name] = column_info


SPARC_TRAIN_DATABASES = [
    "activity_1",
    "aircraft",
    "allergy_1",
    "apartment_rentals",
    "architecture",
    "assets_maintenance",
    "baseball_1",
    "behavior_monitoring",
    "bike_1",
    "body_builder",
    "book_2",
    "browser_web",
    "candidate_poll",
    "chinook_1",
    "cinema",
    "city_record",
    "climbing",
    "club_1",
    "coffee_shop",
    "college_1",
    "college_2",
    "college_3",
    "company_1",
    "company_employee",
    "company_office",
    "county_public_safety",
    "cre_Doc_Control_Systems",
    "cre_Doc_Tracking_DB",
    "cre_Docs_and_Epenses",
    "cre_Drama_Workshop_Groups",
    "cre_Theme_park",
    "csu_1",
    "culture_company",
    "customer_complaints",
    "customer_deliveries",
    "customers_and_addresses",
    "customers_and_invoices",
    "customers_and_products_contacts",
    "customers_campaigns_ecommerce",
    "customers_card_transactions",
    "debate",
    "decoration_competition",
    "department_management",
    "department_store",
    "device",
    "document_management",
    "dorm_1",
    "driving_school",
    "e_government",
    "e_learning",
    "election",
    "election_representative",
    "entertainment_awards",
    "entrepreneur",
    "epinions_1",
    "farm",
    "film_rank",
    "flight_1",
    "flight_4",
    "flight_company",
    "formula_1",
    "game_1",
    "game_injury",
    "gas_company",
    "gymnast",
    "hospital_1",
    "hr_1",
    "icfp_1",
    "inn_1",
    "insurance_and_eClaims",
    "insurance_fnol",
    "insurance_policies",
    "journal_committee",
    "loan_1",
    "local_govt_and_lot",
    "local_govt_in_alabama",
    "local_govt_mdm",
    "machine_repair",
    "manufactory_1",
    "manufacturer",
    "match_season",
    "medicine_enzyme_interaction",
    "mountain_photos",
    "movie_1",
    "music_1",
    "music_2",
    "music_4",
    "musical",
    "network_2",
    "news_report",
    "party_host",
    "party_people",
    "performance_attendance",
    "perpetrator",
    "phone_1",
    "phone_market",
    "pilot_record",
    "product_catalog",
    "products_for_hire",
    "products_gen_characteristics",
    "program_share",
    "protein_institute",
    "race_track",
    "railway",
    "restaurant_1",
    "riding_club",
    "roller_coaster",
    "sakila_1",
    "school_bus",
    "school_finance",
    "school_player",
    "scientist_1",
    "ship_1",
    "ship_mission",
    "shop_membership",
    "small_bank_1",
    "soccer_1",
    "soccer_2",
    "solvency_ii",
    "sports_competition",
    "station_weather",
    "store_1",
    "store_product",
    "storm_record",
    "student_1",
    "student_assessment",
    "swimming",
    "theme_gallery",
    "tracking_grants_for_research",
    "tracking_orders",
    "tracking_share_transactions",
    "tracking_software_problems",
    "train_station",
    "twitter_1",
    "university_basketball",
    "voter_2",
    "wedding",
    "wine_1",
    "workshop_paper",
    "wrestler",
]

SPARC_DEV_DATABASES = [
    "battle_death",
    "car_1",
    "concert_singer",
    "course_teach",
    "cre_Doc_Template_Mgt",
    "dog_kennels",
    "employee_hire_evaluation",
    "flight_2",
    "museum_visit",
    "network_1",
    "orchestra",
    "pets_1",
    "poker_player",
    "real_estate_properties",
    "singer",
    "student_transcripts_tracking",
    "tvshow",
    "voter_1",
    "world_1",
    # 'wta_1' # This one has issues decoding
]
