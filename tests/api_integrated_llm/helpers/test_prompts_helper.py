import json
import os
from pathlib import Path

from api_integrated_llm.data_models.source_models import QuerySourceDataModel
from api_integrated_llm.helpers.instruct_data_prep import get_input_query


project_root_path = Path(__file__).parent.parent.parent.parent.resolve()


def test_get_input_query() -> None:
    prompts_file = os.path.join(project_root_path, "source", "prompts", "prompts.json")
    with open(prompts_file) as f:
        prompt_meta_dict = json.load(f)

    model_names = [
        "granite",
        "llama",
        "hammer",
        "mixtral_8x7b",
        "mixtral-8x22B",
        "deepseek",
        "watt",
        "qwen",
        "gpt",
    ]
    sample_input = "What is your name?"
    example_str = "example"
    function_str = "function"
    sample = QuerySourceDataModel(
        sample_id=0,
        input=sample_input,
        output=[],
        gold_answer=0,
        original_output=[],
        initialization_step=[],
        tools=[],
    )
    for key in ["sequencing", "router"]:
        prompt_dict = prompt_meta_dict[key]
        for model_name in model_names:
            get_input_query(
                sample_input,
                model_name,
                sample,
                example_str,
                prompt_dict,
                function_str,
            )
