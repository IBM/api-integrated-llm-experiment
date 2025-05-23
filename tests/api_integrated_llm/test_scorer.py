import os
from pathlib import Path

from api_integrated_llm.helpers.file_helper import get_files_in_folder
from api_integrated_llm.scoring import parsing, scoring


project_root_path = Path(__file__).parent.parent.parent.resolve()


def test_scorer_llm_evaluator_output() -> None:
    has_exception, _ = scoring(
        evaluator_output_file_paths=get_files_in_folder(  # type: ignore
            folder_path=Path(
                os.path.join(
                    project_root_path,
                    "tests",
                    "data",
                    "test_output",
                    "evaluation",
                    "llm",
                )
            ),
            file_extension="jsonl",
        ),
        output_folder_path=Path(os.path.join(project_root_path, "output", "scoring")),  # type: ignore
    )

    assert not has_exception


def test_scorer_llm_evaluator_output_ignore_file_path_provided() -> None:
    has_exception, num_ignored_samples = scoring(
        evaluator_output_file_paths=get_files_in_folder(  # type: ignore
            folder_path=Path(
                os.path.join(
                    project_root_path,
                    "tests",
                    "data",
                    "test_output",
                    "evaluation",
                    "llm",
                )
            ),
            file_extension="jsonl",
        ),
        output_folder_path=Path(os.path.join(project_root_path, "output", "scoring")),  # type: ignore
        ignore_file_path=Path(
            os.path.join(
                project_root_path,
                "tests",
                "data",
                "source",
                "auxiliary",
                "ignore.json",
            )
        ),
    )

    assert not has_exception
    assert num_ignored_samples == 3


def test_scorer_agent() -> None:
    has_exception, _ = scoring(
        evaluator_output_file_paths=get_files_in_folder(  # type: ignore
            folder_path=Path(
                os.path.join(
                    project_root_path,
                    "tests",
                    "data",
                    "test_output",
                    "parsing",
                )
            ),
            file_extension="json",
        ),
        output_folder_path=Path(os.path.join(project_root_path, "output", "scoring_from_parsing")),  # type: ignore
    )

    assert not has_exception


def test_scorer_with_parser_output() -> None:
    has_exception, _ = scoring(
        evaluator_output_file_paths=get_files_in_folder(  # type: ignore
            folder_path=Path(
                os.path.join(
                    project_root_path,
                    "tests",
                    "data",
                    "test_output",
                    "evaluation",
                    "llm",
                )
            ),
            file_extension="jsonl",
        ),
        output_folder_path=Path(os.path.join(project_root_path, "output", "scoring")),  # type: ignore
        is_single_intent_detection=True,
    )

    assert not has_exception


def test_parser_only_single_intent() -> None:
    has_exception, num_ignored = parsing(
        evaluator_output_file_paths=get_files_in_folder(  # type: ignore
            folder_path=Path(
                os.path.join(
                    project_root_path,
                    "tests",
                    "data",
                    "test_output",
                    "evaluation_single_intent",
                    "llm",
                )
            ),
            file_extension="jsonl",
        ),
        output_folder_path=Path(os.path.join(project_root_path, "output", "parsing")),  # type: ignore
        is_single_intent_detection=True,
        ignore_file_path=None,
    )

    assert not has_exception
    assert num_ignored == 0


def test_parser_only_regular_sequence() -> None:
    has_exception, num_ignored = parsing(
        evaluator_output_file_paths=get_files_in_folder(  # type: ignore
            folder_path=Path(
                os.path.join(
                    project_root_path,
                    "tests",
                    "data",
                    "test_output",
                    "evaluation_regular_sequence",
                    "llm",
                )
            ),
            file_extension="jsonl",
        ),
        output_folder_path=Path(os.path.join(project_root_path, "output", "parsing")),  # type: ignore
        is_single_intent_detection=False,
        ignore_file_path=None,
    )

    assert not has_exception
    assert num_ignored == 0
