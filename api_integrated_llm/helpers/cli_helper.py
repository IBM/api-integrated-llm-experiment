import argparse
from pathlib import Path
import os
from typing import Any, Dict
from api_integrated_llm.evaluation import evaluate
from api_integrated_llm.helpers.benchmark_helper import get_model_id_obj_dict
from api_integrated_llm.helpers.file_helper import (
    get_dict_from_json,
    get_files_in_folder,
)
from api_integrated_llm.scoring import scoring


def get_arguments() -> argparse.Namespace:
    project_root_path = Path(__file__).parent.parent.parent.resolve()
    parser = argparse.ArgumentParser(description="Conversational AI Gym Tool")

    # Common arguments
    parser.add_argument(
        "-rt",
        "--root",
        type=str,
        help="Dataset root absolute path",
        default=str(project_root_path),
    )

    return parser.parse_args()


def get_llm_configuration(llm_configuration_file_path: Path) -> Dict[str, Any]:
    return (
        get_dict_from_json(file_path=llm_configuration_file_path)
        if os.path.isfile((llm_configuration_file_path))
        else get_model_id_obj_dict()
    )


def cli() -> None:
    args = get_arguments()
    source_folder_path = os.path.join(args.root, "source")
    output_folder_path = os.path.join(args.root, "output")
    evaluation_folder_path = os.path.join(
        output_folder_path,
        "evaluation",
    )

    evaluate(
        model_id_info_dict=(
            get_llm_configuration(
                llm_configuration_file_path=os.path.join(
                    source_folder_path, "configurations", "llm_configurations.json"
                )
            )
        ),
        evaluation_input_file_paths=get_files_in_folder(
            folder_path=os.path.join(source_folder_path, "evaluation"),
            file_extension="json",
        ),
        example_file_path=os.path.join(
            source_folder_path, "prompts", "examples_icl.json"
        ),
        output_folder_path=evaluation_folder_path,
        prompt_file_path=os.path.join(source_folder_path, "prompts", "prompts.json"),
        error_folder_path=os.path.join(
            output_folder_path,
            "error",
        ),
        temperatures=[0.0],
        max_tokens_list=[1500],
        should_generate_random_example=True,
        num_examples=3,
    )
    scoring(
        evaluator_output_file_paths=get_files_in_folder(
            folder_path=evaluation_folder_path,
            file_extension="jsonl",
        ),
        output_folder_path=os.path.join(output_folder_path, "scoring"),
        win_rate_flag=False,
    )
