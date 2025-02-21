import os
from pathlib import Path

from api_integrated_llm.helpers.file_helper import get_list_dict_from_jsonl
from api_integrated_llm.helpers.output_parsers import parse_llama_3_output


test_root_path = Path(__file__).parent.parent.parent.resolve()


def test_parse_llama_3_output_single_intent() -> None:
    data_list = get_list_dict_from_jsonl(
        file_path=Path(
            os.path.join(
                test_root_path,
                "data",
                "output",
                "evaluation",
                "llama",
                "real_estate_properties_nestful_format_cosql.jsonl",
            )
        ),
    )

    (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
    ) = parse_llama_3_output(
        prediction=data_list[0],
        num_errors_parsing_pred_intent=0,
        skip_grounding=False,
        is_single_intent_detection=True,
    )

    assert not pred_has_parsing_errors


def test_parse_llama_3_output_multi_intent() -> None:
    data_list = get_list_dict_from_jsonl(
        file_path=Path(
            os.path.join(
                test_root_path,
                "data",
                "output",
                "evaluation",
                "llama",
                "real_estate_properties_nestful_format_cosql.jsonl",
            )
        ),
    )

    (
        pred_func_calls,
        gold_func_calls,
        pred_dict_list,
        gold_dict_list,
        num_errors_parsing_pred_intent,
        pred_has_parsing_errors,
    ) = parse_llama_3_output(
        prediction=data_list[0],
        num_errors_parsing_pred_intent=0,
        skip_grounding=False,
        is_single_intent_detection=False,
    )

    assert not pred_has_parsing_errors
