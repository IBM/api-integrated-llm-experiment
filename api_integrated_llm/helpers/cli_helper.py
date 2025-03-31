import argparse
from pathlib import Path
import os
from typing import Any, Dict, Optional, Tuple, cast
from api_integrated_llm.data_models.cli_models import CliModeModel
from api_integrated_llm.evaluation import evaluate
from api_integrated_llm.helpers.file_helper import (
    get_date_time_str,
    get_dict_from_json,
    get_files_in_folder,
)
from api_integrated_llm.metrics_aggregator import aggregate_metrics
from api_integrated_llm.scoring import parsing, scoring


project_root_path = Path(__file__).parent.parent.parent.resolve()


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="API Integrated LLM CLI")

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
        help="Dataset root folder absolute path",
        default=project_root_path,
    )

    parser.add_argument(
        "-eof",
        "--evaluator_output_folder",
        type=Path,
        help="Evaluator output folder path",
        default=project_root_path,
    )

    parser.add_argument(
        "-pif",
        "--parser_input_folder",
        type=Path,
        help="Parser input folder path",
        default=project_root_path,
    )

    parser.add_argument(
        "-of",
        "--output_folder",
        type=Path,
        help="Output folder path",
        default=project_root_path,
    )

    parser.add_argument(
        "-sif",
        "--scorer_input_folder",
        type=Path,
        help="Scorer input folder path",
        default=project_root_path,
    )

    parser.add_argument(
        "-maif",
        "--metrics_aggregator_input_folder",
        type=Path,
        help="Metrics aggregator input folder path",
        default=project_root_path,
    )

    parser.add_argument(
        "-ig",
        "--ignore",
        default=True,
        action=argparse.BooleanOptionalAction,
        help='Ignore data points marked as "ignore"',
    )

    parser.add_argument(
        "-igf",
        "--ignore_file_path",
        type=Path,
        help="Ignore file path",
        default=project_root_path,
    )

    parser.add_argument(
        "-asy",
        "--use_async",
        action=argparse.BooleanOptionalAction,
        help="Use asynchronous operations",
    )

    parser.add_argument(
        "-si",
        "--single_intent",
        action=argparse.BooleanOptionalAction,
        help="Single intent dataset",
    )

    parser.add_argument(
        "-atp",
        "--add_tool_definition_to_prompt",
        action=argparse.BooleanOptionalAction,
        help="Add tool definitions to prompt",
    )

    parser.add_argument(
        "-er",
        "--random_example",
        action=argparse.BooleanOptionalAction,
        help="Create examples in prompts by sampling source data randomly",
    )

    parser.add_argument(
        "-nr",
        "--number_random_example",
        type=int,
        help="The number of examples sampled from source data randomly",
        default=2,
    )

    parser.add_argument(
        "-ep",
        "--example_file_path",
        type=Path,
        help="The absolute path for an example file for a prompt",
        default=project_root_path,
    )

    parser.add_argument(
        "-mcp",
        "--language_model_configuration_file_path",
        type=Path,
        help="The absolute path for a language model configuration",
        default=project_root_path,
    )

    return parser.parse_args()


def get_llm_configuration(llm_configuration_file_path: Path) -> Dict[str, Any]:
    return (
        get_dict_from_json(file_path=llm_configuration_file_path)  # type: ignore
        if os.path.isfile((llm_configuration_file_path))
        else {}
    )


def check_args(args) -> bool:
    should_stop = False
    if args.mode == CliModeModel.SCORER:
        if args.output_folder == project_root_path:
            print("output_folder should be defined")
            should_stop = True
        if args.scorer_input_folder == project_root_path:
            print("scorer input folder_folder should be defined")
            should_stop = True
    elif args.mode == CliModeModel.PARSER:
        if args.output_folder == project_root_path:
            print("output_folder should be defined")
            should_stop = True
        if args.parser_input_folder == project_root_path:
            print("parser input folder should be defined")
            should_stop = True
    elif args.mode == CliModeModel.METRICS_AGGREGATOR:
        if args.output_folder == project_root_path:
            print("output_folder should be defined")
            should_stop = True
        if args.metrics_aggregator_input_folder == project_root_path:
            print("metrics aggregator input folder should be defined")
            should_stop = True

    return should_stop


def get_paths(
    args,
) -> Tuple[Path, Path, Path, Path, Path, Path, Optional[Path]]:
    root_folder_path = Path(os.path.join(args.root, "source"))
    output_folder_path = (
        Path(os.path.join(args.root, "output"))
        if args.output_folder == project_root_path
        else args.output_folder
    )
    evaluation_folder_path = (
        Path(
            os.path.join(
                output_folder_path,
                "evaluation",
            )
        )
        if args.evaluator_output_folder == project_root_path
        else args.evaluator_output_folder
    )

    parser_input_folder_path = (
        Path(
            os.path.join(
                output_folder_path,
                "evaluation",
            )
        )
        if args.parser_input_folder == project_root_path
        else args.parser_input_folder
    )

    scorer_input_folder_path = (
        Path(
            os.path.join(
                output_folder_path,
                "evaluation",
            )
        )
        if args.scorer_input_folder == project_root_path
        else args.scorer_input_folder
    )

    metrics_aggregator_input_folder_path = (
        Path(
            os.path.join(
                output_folder_path,
                "scoring",
            )
        )
        if args.metrics_aggregator_input_folder == project_root_path
        else args.metrics_aggregator_input_folder
    )

    ignore_file_path = (
        None
        if args.ignore_file_path == project_root_path
        else cast(Path, args.ignore_file_path)
    )

    return (
        root_folder_path,
        cast(Path, output_folder_path),
        cast(Path, evaluation_folder_path),
        cast(Path, parser_input_folder_path),
        cast(Path, scorer_input_folder_path),
        metrics_aggregator_input_folder_path,
        ignore_file_path,
    )


def cli() -> None:
    args = get_arguments()

    if check_args(args):
        return

    (
        root_folder_path,
        output_folder_path,
        evaluation_folder_path,
        parser_input_folder_path,
        scorer_input_folder_path,
        metrics_aggregator_input_folder_path,
        ignore_file_path,
    ) = get_paths(args)

    if args.mode == CliModeModel.DEFAULT or args.mode == CliModeModel.EVALUATOR:
        evaluate(
            model_id_info_dict=(
                get_llm_configuration(
                    llm_configuration_file_path=Path(
                        os.path.join(
                            root_folder_path,
                            "configurations",
                            "llm_configurations.json",
                        )
                        if (
                            args.language_model_configuration_file_path
                            == project_root_path
                        )
                        else args.language_model_configuration_file_path
                    )
                )
            ),
            evaluation_input_file_paths=get_files_in_folder(
                folder_path=Path(os.path.join(root_folder_path, "evaluation")),
                file_extension="json",
            ),
            example_file_path=(
                Path(
                    os.path.join(
                        root_folder_path, "prompts", "examples", "examples.json"
                    )
                )
                if args.example_file_path == project_root_path
                else args.example_file_path
            ),
            output_folder_path=evaluation_folder_path,
            prompt_file_path=Path(
                os.path.join(root_folder_path, "prompts", "prompts.json")
            ),
            error_folder_path=Path(
                os.path.join(
                    output_folder_path,
                    "error",
                )
            ),
            temperatures=[0.0],
            max_tokens_list=[1500],
            should_generate_random_example=args.random_example,
            num_examples=args.number_random_example,
            should_ignore=args.ignore,
            should_async=args.use_async,
            should_add_tool_definitions_to_prompt=args.add_tool_definition_to_prompt,
        )

    if args.mode == CliModeModel.PARSER:
        print("\n")
        print(f"Parser input folder: {parser_input_folder_path}")
        print(f"Parser output folder: {output_folder_path}")
        print("\n")
        parsing(
            evaluator_output_file_paths=get_files_in_folder(  # type: ignore
                folder_path=parser_input_folder_path,
                file_extension="jsonl",
            ),
            output_folder_path=Path(os.path.join(output_folder_path, "parsing")),
            is_single_intent_detection=args.single_intent,
            ignore_file_path=ignore_file_path,
        )

    if args.mode == CliModeModel.DEFAULT or args.mode == CliModeModel.SCORER:
        print("\n")
        print(f"Scorer input folder: {scorer_input_folder_path}")
        print(f"Scorer output folder: {output_folder_path}")
        print("\n")
        scoring(
            evaluator_output_file_paths=get_files_in_folder(  # type: ignore
                folder_path=scorer_input_folder_path,
                file_extension="jsonl",
            ),
            output_folder_path=Path(os.path.join(output_folder_path, "scoring")),
            is_single_intent_detection=args.single_intent,
            ignore_file_path=ignore_file_path,
        )
    if (
        args.mode == CliModeModel.DEFAULT
        or args.mode == CliModeModel.METRICS_AGGREGATOR
    ):
        print("\n")
        print(
            f"Metrics aggregator input folder: {metrics_aggregator_input_folder_path}"
        )
        print(f"Scorer output folder: {output_folder_path}")
        print("\n")
        time_str = get_date_time_str()
        aggregate_metrics(
            metrics_aggregator_input_path=metrics_aggregator_input_folder_path,
            metrics_aggregator_output_file_path=Path(
                os.path.join(
                    output_folder_path,
                    "metrics_aggregation",
                    f"metrics_aggregation_{time_str}.json",
                )
            ),
            metrics_aggregation_configuration_file_path=None,
        )
