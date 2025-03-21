import os
from pathlib import Path

from api_integrated_llm.helpers.file_helper import get_list_dict_from_jsonl
from api_integrated_llm.helpers.output_parsers import (
    parse_general_large_language_model_output,
    parse_multi_step,
)


test_root_path = Path(__file__).parent.parent.parent.resolve()


def test_parse_llama_3_output_single_intent() -> None:
    data_list = get_list_dict_from_jsonl(
        file_path=Path(
            os.path.join(
                test_root_path,
                "data",
                "test_output",
                "evaluation",
                "llm",
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
        parsing_error_messages,
    ) = parse_general_large_language_model_output(
        prediction=data_list[0],
        num_errors_parsing_pred_intent=0,
        skip_grounding=False,
    )

    assert not pred_has_parsing_errors


def test_parse_llama_3_output_multi_intent() -> None:
    data_list = get_list_dict_from_jsonl(
        file_path=Path(
            os.path.join(
                test_root_path,
                "data",
                "test_output",
                "evaluation",
                "llm",
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
        parsing_error_messages,
    ) = parse_general_large_language_model_output(
        prediction=data_list[0],
        num_errors_parsing_pred_intent=0,
        skip_grounding=False,
    )

    assert not pred_has_parsing_errors


def test_parse_multi_step() -> None:
    txt = '[select_data_less_than_equal_to(data_source="$starting_table_var$", key_name="edhrecRank", value=100)]'
    res = parse_multi_step(txt=txt)

    assert isinstance(res, list)
    assert len(res) == 1
