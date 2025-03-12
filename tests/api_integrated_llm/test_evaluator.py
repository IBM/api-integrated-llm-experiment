import os
from pathlib import Path

import pytest

from api_integrated_llm.helpers.file_helper import get_files_in_folder
from api_integrated_llm.evaluation import evaluate


project_root_path = Path(__file__).parent.parent.parent.resolve()


@pytest.mark.skip(reason="sanity check for transformer package")
def test_evaluator_local_llm() -> None:
    model_id_info_dict = {
        "smol_llama-101M-GQA": {
            "inference_type": "LOCAL",
            "model": "BEE-spoke-data/smol_llama-101M-GQA",
            "endpoint": "BEE-spoke-data/smol_llama-101M-GQA",
            "tokenizer": "BEE-spoke-data/smol_llama-101M-GQA",
            "should_use_autoprocessor": False,
        }
    }

    evaluate(
        model_id_info_dict=model_id_info_dict,
        evaluation_input_file_paths=get_files_in_folder(  # type: ignore
            folder_path=Path(
                os.path.join(project_root_path, "tests", "data", "source", "evaluation")
            ),
            file_extension="json",
        ),
        example_file_path=Path(
            os.path.join(
                project_root_path,
                "tests",
                "data",
                "source",
                "prompts",
                "examples",
                "examples.json",
            )
        ),
        output_folder_path=Path(
            os.path.join(project_root_path, "output", "evaluation")
        ),
        prompt_file_path=Path(
            os.path.join(
                project_root_path, "tests", "data", "source", "prompts", "prompts.json"
            )
        ),
        error_folder_path=Path(os.path.join(project_root_path, "output", "error")),
        temperatures=[0.0],
        max_tokens_list=[1500],
        should_generate_random_example=False,
        num_examples=1,
        should_ignore=True,
        should_async=False,
    )

    assert True
