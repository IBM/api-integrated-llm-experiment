import os
from pathlib import Path

from api_integrated_llm.data_models.source_models import QuerySourceModel
from api_integrated_llm.helpers.file_helper import (
    get_base_model_from_json,
    get_files_in_folder,
    write_json,
)


project_root_path = Path(__file__).parent.parent.parent.parent.resolve()


def test_parse_sources() -> None:
    for file_path in get_files_in_folder(
        folder_path=Path(
            os.path.join(project_root_path, "tests", "data", "source", "evaluation")
        ),
        file_extension="json",
    ):
        _ = get_base_model_from_json(
            file_path=file_path,
            base_model=QuerySourceModel,
        )
    assert True


def test_parse_sources_add_sample_id() -> None:
    obj = get_base_model_from_json(
        file_path=Path(
            os.path.join(
                project_root_path,
                "tests",
                "data",
                "source",
                "evaluation",
                "sequencing_bird_superhero.json",
            )
        ),
        base_model=QuerySourceModel,
    )

    obj.dataset = "superhero"
    for i in range(len(obj.data)):
        obj.data[i].sample_id = i + 1

    write_json(
        file_path=Path(
            os.path.join(
                project_root_path,
                "tests",
                "data",
                "source",
                "evaluation",
                "sequencing_bird_superhero.json",
            )
        ),
        base_model=obj,
    )

    assert True


def test_parse_prompt_templates_examples() -> None:
    for file_path in get_files_in_folder(
        folder_path=Path(os.path.join(project_root_path, "source", "prompts")),
        file_extension="json",
    ):
        _ = get_base_model_from_json(
            file_path=file_path,
            base_model=QuerySourceModel,
        )
    assert True
