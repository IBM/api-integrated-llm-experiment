import os
from pathlib import Path

from api_integrated_llm.helpers.file_helper import get_files_in_folder
from api_integrated_llm.scoring import scoring


project_root_path = Path(__file__).parent.parent.parent.resolve()


def test_scorer() -> None:
    scoring(
        evaluator_output_file_paths=get_files_in_folder(  # type: ignore
            folder_path=Path(
                os.path.join(project_root_path, "tests", "data", "output", "evaluation")
            ),
            file_extension="jsonl",
        ),
        output_folder_path=Path(os.path.join(project_root_path, "output", "scoring")),  # type: ignore
        win_rate_flag=False,
    )
