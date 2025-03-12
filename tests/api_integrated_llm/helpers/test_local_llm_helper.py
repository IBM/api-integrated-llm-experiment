import pytest
from api_integrated_llm.helpers.database_helper.local_llm_helper import (
    get_response_from_llm_with_tokenizer,
)


@pytest.mark.skip(reason="sanity check for transformer package")
def test_local_llm() -> None:
    model_obj = {
        "inference_type": "LOCAL",
        "model": "BEE-spoke-data/smol_llama-101M-GQA",
        "endpoint": "BEE-spoke-data/smol_llama-101M-GQA",
        "tokenizer": "BEE-spoke-data/smol_llama-101M-GQA",
        "should_use_autoprocessor": False,
    }
    input = "hello!"
    response = get_response_from_llm_with_tokenizer(
        input=input, model_obj=model_obj, temperature=0.0, max_tokens=100
    )
    assert response is not None
    assert len(response) > 0
