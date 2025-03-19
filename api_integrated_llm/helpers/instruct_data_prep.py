from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


from api_integrated_llm.data_models.source_models import (
    ConversationRoleModel,
    ConversationUnit,
    DataUnit,
    EvaluationOutputDataUnit,
    ExampleDataModel,
    QuerySourceDataModel,
    QuerySourceModel,
)
from api_integrated_llm.helpers.file_helper import (
    get_base_model_from_json,
    get_dict_from_json,
)
from api_integrated_llm.helpers.sampling_helper import get_random_example_for_prompt
from api_integrated_llm.helpers.tokenizer_helper import granite_prompt_input


def get_example_str(icl_examples: List[DataUnit]) -> str:
    example_strs: list[str] = []
    counter = 1
    for ex in icl_examples:
        if ex.output is not None:
            tmp = [item.model_dump() for item in ex.output]
            example_strs.append(
                f"\n#Example-{counter}\nInput: {ex.input}\nOutput: {json.dumps(tmp)}\n"
            )
            counter += 1
    return "".join(example_strs)


def get_examples(
    example_file_path: Path,
    evaluation_input_file_paths: List[Path],
    chosen_evaluation_input_file_path: Path,
    num_examples: int,
    should_generate_random_example: bool,
) -> List[DataUnit]:
    if should_generate_random_example:
        return get_random_example_for_prompt(
            evaluation_input_file_paths=evaluation_input_file_paths,  # type: ignore
            chosen_evaluation_input_file_path=chosen_evaluation_input_file_path,
            num_examples=num_examples,
        )

    example_model: ExampleDataModel = get_base_model_from_json(
        example_file_path, ExampleDataModel
    )

    return example_model.data


def get_prompt_dict(
    prompt_file_path: Path, evaluation_input_file_path: Path
) -> Dict[str, str]:
    prompt_dict_all = get_dict_from_json(file_path=prompt_file_path)  # type: ignore
    path_str = str(evaluation_input_file_path)
    if "rest" in path_str:
        return prompt_dict_all["router"]
    if "sequencing" in path_str or "slot_filling" in path_str:
        return prompt_dict_all["sequencing"]
    return prompt_dict_all["sequencing"]


def replace_sub_str(
    base_str: str, function_str: str, example_str: str, sample_input: str
) -> str:
    pairs = {
        "{FUNCTION_STR}": function_str,
        "{ICL_EXAMPLES}": example_str,
        "{QUERY}": sample_input,
    }
    new_str = deepcopy(base_str)
    for key, val in pairs.items():
        new_str = new_str.replace(key, val)

    return new_str


def get_input_query(
    sample_input: str,
    model_name: str,
    sample: QuerySourceDataModel,
    example_str: str,
    prompt_dict: Dict[str, str],
    function_str: str,
) -> str:
    model_name_lower = model_name.lower()
    if "granite" in model_name_lower:
        return granite_prompt_input(
            sample_input,
            (sample.tools if sample.tools is not None else []),
            example_str,
            prompt_dict["granite"],
        )

    template_str = ""

    if "hammer" in model_name_lower and "Hammer2.0-7b" in prompt_dict:
        template_str = prompt_dict["Hammer2.0-7b"]
    elif (
        "mixtral_8x7b" in model_name_lower
        and "mixtral_8x7b_instruct_v01" in prompt_dict
    ):
        template_str = prompt_dict["mixtral_8x7b_instruct_v01"]
    elif (
        "mixtral-8x22B" in model_name_lower
        and "Mixtral-8x22B-Instruct-v0.1" in prompt_dict
    ):
        template_str = prompt_dict["Mixtral-8x22B-Instruct-v0.1"]
    elif "deepseek" in model_name_lower and "DeepSeek-V3" in prompt_dict:
        template_str = prompt_dict["DeepSeek-V3"]
    elif "watt" in model_name_lower and "watt-tool-8B" in prompt_dict:
        template_str = prompt_dict["watt-tool-8B"]
    elif "qwen" in model_name_lower and "Qwen2.5-7B-Instruct" in prompt_dict:
        template_str = prompt_dict["Qwen2.5-7B-Instruct"]
    elif "llama" in model_name_lower:
        template_str = prompt_dict["LLaMa-3.1"]
    elif "gpt" in model_name_lower:
        template_str = prompt_dict["gpt4o"]
    else:
        template_str = prompt_dict["LLaMa-3.1"]

    return replace_sub_str(
        base_str=template_str,
        function_str=function_str,
        example_str=example_str,
        sample_input=sample_input,
    )


def get_OPENAI_messages(
    system_prompt: str, sample_input: str, tools: Optional[List[Dict[str, Any]]]
) -> List[ConversationUnit]:
    tool_str = (
        ("\n\nAvailable Tools:\n" + json.dumps(tools)) if tools is not None else ""
    )
    system_utterance = ConversationUnit(
        role=ConversationRoleModel.SYSTEM,
        content=(
            system_prompt
            if "Available Tools:" in system_prompt
            else (system_prompt + tool_str)
        ),
    )
    user_utterance = ConversationUnit(
        role=ConversationRoleModel.USER, content=sample_input
    )
    return [system_utterance, user_utterance]


def instruct_data(
    prompt_file_path: Path,
    model_name: str,
    model_obj: Dict[str, Any],
    evaluation_input_file_path: Path,
    evaluation_input_file_paths: List[Path],
    example_file_path: Optional[Path] = None,
    should_generate_random_example: bool = False,
    num_examples: int = 1,
    should_ignore: bool = True,
) -> Tuple[List[EvaluationOutputDataUnit], Optional[str]]:
    examples = get_examples(
        example_file_path=example_file_path,  # type: ignore
        evaluation_input_file_paths=evaluation_input_file_paths,  # type: ignore
        chosen_evaluation_input_file_path=evaluation_input_file_path,
        num_examples=num_examples,
        should_generate_random_example=should_generate_random_example,
    )

    if len(examples) == 0:
        raise Exception("No example data is found.")

    prompt_dict = get_prompt_dict(
        prompt_file_path=prompt_file_path,
        evaluation_input_file_path=evaluation_input_file_path,
    )

    source_model: QuerySourceModel = get_base_model_from_json(
        file_path=evaluation_input_file_path,
        base_model=QuerySourceModel,
    )
    dataset = source_model.dataset
    test_data: List[EvaluationOutputDataUnit] = []
    example_str = get_example_str(examples)

    if source_model.data is None:
        return test_data, None

    for sample in source_model.data:
        if should_ignore and sample.ignore is not None and sample.ignore:
            continue

        function_str = (
            json.dumps(list((map(lambda item: item.model_dump(), sample.tools))))
            if sample.tools is not None
            else ""
        )
        sample_input = sample.input if sample.input is not None else ""
        input_query = get_input_query(
            sample_input=sample_input,
            model_name=model_name,
            sample=sample,
            example_str=example_str,
            prompt_dict=prompt_dict,
            function_str=function_str,
        )
        tools = sample.get_tools_raw()
        test_data.append(
            EvaluationOutputDataUnit(
                sample_id=sample.sample_id,
                input=input_query,
                output=sample.output,
                gold_answer=sample.gold_answer,
                messages=(
                    get_OPENAI_messages(
                        system_prompt=input_query,
                        sample_input=sample_input,
                        tools=tools,
                    )
                    if model_obj.get("inference_type", "") == "OPENAI"
                    else None
                ),
                tools=tools,
            )
        )

    return test_data, dataset
