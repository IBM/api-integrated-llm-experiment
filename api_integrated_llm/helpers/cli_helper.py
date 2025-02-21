import argparse
from pathlib import Path
import os
from typing import Any, Dict
from api_integrated_llm.data_models.cli_models import CliModeModel
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

    parser.add_argument(
        "-m",
        "--mode",
        type=CliModeModel,
        help="Cli mode",
        default=CliModeModel.DEFAULT,
        choices=list(CliModeModel),
    )

    # Common arguments
    parser.add_argument(
        "-rt",
        "--root",
        type=Path,
        help="Dataset root absolute path",
        default=project_root_path,
    )

    return parser.parse_args()


def get_llm_configuration(llm_configuration_file_path: Path) -> Dict[str, Any]:
    return (
        get_dict_from_json(file_path=llm_configuration_file_path)  # type: ignore
        if os.path.isfile((llm_configuration_file_path))
        else get_model_id_obj_dict()
    )


def cli() -> None:
    args = get_arguments()
    source_folder_path = Path(os.path.join(args.root, "source"))
    output_folder_path = Path(os.path.join(args.root, "output"))
    evaluation_folder_path = Path(
        os.path.join(
            output_folder_path,
            "evaluation",
        )
    )
    if args.mode == CliModeModel.DEFAULT or args.mode == CliModeModel.EVALUATOR:
        evaluate(
            model_id_info_dict=(
                get_llm_configuration(
                    llm_configuration_file_path=os.path.join(  # type: ignore
                        source_folder_path, "configurations", "llm_configurations.json"
                    )
                )
            ),
            evaluation_input_file_paths=get_files_in_folder(
                folder_path=Path(os.path.join(source_folder_path, "evaluation")),
                file_extension="json",
            ),
            example_file_path=os.path.join(  # type: ignore
                source_folder_path, "prompts", "examples_icl.json"
            ),
            output_folder_path=evaluation_folder_path,  # type: ignore
            prompt_file_path=os.path.join(source_folder_path, "prompts", "prompts.json"),  # type: ignore
            error_folder_path=os.path.join(  # type: ignore
                output_folder_path,
                "error",
            ),
            temperatures=[0.0],
            max_tokens_list=[1500],
            should_generate_random_example=True,
            num_examples=3,
        )

    if args.mode == CliModeModel.DEFAULT or args.mode == CliModeModel.SCORER:
        scoring(
            evaluator_output_file_paths=get_files_in_folder(  # type: ignore
                folder_path=evaluation_folder_path,
                file_extension="jsonl",
            ),
            output_folder_path=Path(os.path.join(output_folder_path, "scoring")),  # type: ignore
            win_rate_flag=False,
        )
