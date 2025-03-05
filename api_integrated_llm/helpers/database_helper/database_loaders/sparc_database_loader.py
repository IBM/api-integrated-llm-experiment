from collections import defaultdict
from datetime import datetime
from numpy import isnan
import json
import os

from api_integrated_llm.helpers.database_helper.tools.sql_query_components import (
    make_safe,
)
from api_integrated_llm.helpers.database_helper.database_loaders.database_loader import (
    DatabaseLoader,
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


class SparcDatabaseLoader(DatabaseLoader):
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
            self.database_location,
            "database_preprocessed",
            self.name,
            self.name + ".sqlite",
        )
        assert os.path.isfile(
            self.database_path
        ), f"Database: {self.database_path} was not found. "

    def _load_keys(self, key_data: dict):
        tables = key_data["table_names"]
        columns = key_data["column_names_preprocessed"]

        # Primary and foreign keys are encoded by their index in the total list of database columns
        # Need to translate this global index into a table-column name pair.
        primary_keys = key_data["primary_keys"]
        foreign_keys = key_data["foreign_keys"]

        # Note that a primary key may be composed of multiple columns
        for p in primary_keys:
            col = columns[p]
            col_name = make_safe(col[1])
            table_id = col[0]
            tab = tables[table_id]
            self.primary_keys[tab].append(
                [col_name]
            )  # Make it a list to match format with possible multi-column keys

        for f in foreign_keys:
            col1 = make_safe(columns[f[0]][1])
            tab1 = tables[columns[f[0]][0]]

            col2 = make_safe(columns[f[1]][1])
            tab2 = tables[columns[f[1]][0]]
            self.foreign_keys.append(
                [{"table": tab1, "column": col1}, {"table": tab2, "column": col2}]
            )

    def load_lazy(self):
        # Load individual table descriptions
        table_description_file = os.path.join(
            self.database_location, "tables_preprocessed.json"
        )
        assert os.path.isfile(
            table_description_file
        ), f"{table_description_file} does not exist. "
        with open(table_description_file, "r") as f:
            table_descriptions = json.load(f)

        table = None
        for t in table_descriptions:
            if t["db_id"] == self.name:
                table = t
                break
        else:
            raise Exception(
                f"Table {self.name} not found in {table_description_file}. "
            )

        table_names = table["table_names_preprocessed"]
        table_metadata = defaultdict(list)
        col_name_list = []
        for col, description, format in zip(
            table["column_names_preprocessed"],
            table["column_descriptions"],
            table["column_types"],
        ):
            table_index = col[0]
            if table_index == -1:
                continue

            if isinstance(format, str):
                format = format.strip()
            dtype = dtype_translator_sparc.get(format, None)
            if dtype is None:
                try:
                    if isnan(format):
                        dtype = float
                except:
                    dtype = object

            col_name = col[1]
            col_name_list.append(col_name)
            safe_name = make_safe(col_name)
            metadata = {
                "column_name": safe_name,
                "column_description": description,
                "column_dtype": dtype,
            }
            table_metadata[table_names[table_index]].append(metadata)
        assert set(table_metadata.keys()) == set(
            table_names
        ), f"Missing metadata from tables {set(table_names)}, only found tables {set(table_metadata.keys())}"
        for t in table_names:
            self.table_descriptions[t] = table_metadata[t]
        self.column_list = list(set(col_name_list))  # Only unique column names
        self._load_keys(table)


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
