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
            "model_url": "/Users/jungkookang/Documents/projects/api_integrated_llm_experiment/llm_saves/smol_llama-101M-GQA",  # pragma: allowlist secret
            "tokenizer_url": "/Users/jungkookang/Documents/projects/api_integrated_llm_experiment/tokenizer_saves/smol_llama-101M-GQA",  # pragma: allowlist secret
        }
    }

    evaluate(
        model_id_info_dict=model_id_info_dict,
        evaluation_input_file_paths=get_files_in_folder(  # type: ignore
            folder_path=Path(
                os.path.join(
                    project_root_path, "tests", "data", "source", "evaluation_trimmed"
                )
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
        max_tokens_list=[100],
        should_generate_random_example=False,
        num_examples=1,
        should_ignore=True,
        should_async=False,
    )

    assert True


@pytest.mark.skip(reason="sanity check for RITZ")
def test_evaluator_RITZ() -> None:
    model_id_info_dict = {
        "Llama-3.1-8B-Instruct": {
            "inference_type": "RITS",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-1-8b-instruct/v1",
        },
    }

    evaluate(
        model_id_info_dict=model_id_info_dict,
        evaluation_input_file_paths=get_files_in_folder(  # type: ignore
            folder_path=Path(
                os.path.join(
                    project_root_path, "tests", "data", "source", "evaluation_trimmed"
                )
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
                "examples_icl_sequencing.json",
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
        should_async=True,
    )

    assert True


@pytest.mark.skip(reason="sanity check for OPENAI")
def test_evaluator_openai() -> None:
    model_id_info_dict = {
        "gpt-4o": {
            "inference_type": "OPENAI",
            "model": "gpt-4o",
            "model-id": "gpt-4o-2024-08-06",
            "endpoint": "https://eteopenai.azure-api.net/openai/deployments/{MODEL_ID}/chat/completions",
            "api-version": "2024-08-01-preview",
        }
    }

    evaluate(
        model_id_info_dict=model_id_info_dict,
        evaluation_input_file_paths=get_files_in_folder(  # type: ignore
            folder_path=Path(
                os.path.join(
                    project_root_path,
                    "tests",
                    "data",
                    "source",
                    "evaluation_test",
                )
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
                "examples_icl_sequencing.json",
            )
        ),
        output_folder_path=Path(
            os.path.join(project_root_path, "output", "evaluation")
        ),
        prompt_file_path=Path(
            os.path.join(project_root_path, "source", "prompts", "prompts.json")
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
