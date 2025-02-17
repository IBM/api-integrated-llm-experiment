import os
from pathlib import Path
from api_integrated_llm.evaluation import evaluate
from api_integrated_llm.helpers.benchmark_helper import get_model_id_obj_dict
from api_integrated_llm.helpers.file_helper import get_files_in_folder
from api_integrated_llm.scoring import scoring

current_folder_path = Path(__file__).parent.resolve()

if __name__ == "__main__":
    evaluate(
        model_id_info_dict=get_model_id_obj_dict(),
        evaluation_input_file_paths=get_files_in_folder(
            folder_path=os.path.join(current_folder_path, "source", "evaluation"),
            file_extension="json",
        ),
        example_file_path=os.path.join(
            current_folder_path, "source", "prompts", "examples_icl.json"
        ),
        output_folder_path=os.path.join(
            current_folder_path,
            "output",
            "evaluation",
        ),
        prompt_file_path=os.path.join(
            current_folder_path, "source", "prompts", "prompts.json"
        ),
        error_folder_path=os.path.join(
            current_folder_path,
            "error",
        ),
        temperatures=[0.0],
        max_tokens_list=[1500],
        should_generate_random_example=True,
        num_examples=3,
    )
    scoring(
        evaluator_output_file_paths=get_files_in_folder(
            folder_path=os.path.join(current_folder_path, "output", "evaluation"),
            file_extension="jsonl",
        ),
        output_folder_path=os.path.join(current_folder_path, "output", "scoring"),
        win_rate_flag=False,
    )
