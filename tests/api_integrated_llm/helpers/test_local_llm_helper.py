import os
from pathlib import Path
import pytest
from api_integrated_llm.helpers.database_helper.local_llm_helper import (
    get_response_from_llm_with_tokenizer,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import torch

LLM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
project_root_path = Path(__file__).parent.parent.parent.parent.resolve()


@pytest.mark.skip(reason="sanity check for transformer package")
def test_local_llm() -> None:
    model_obj = {
        "inference_type": "LOCAL",
        "model": "BEE-spoke-data/smol_llama-101M-GQA",
        "endpoint": "BEE-spoke-data/smol_llama-101M-GQA",
        "tokenizer": "BEE-spoke-data/smol_llama-101M-GQA",
        "model_url": "/Users/jungkookang/Documents/projects/api_integrated_llm_experiment/llm_saves/smol_llama-101M-GQA",  # pragma: allowlist secret
        "tokenizer_url": "/Users/jungkookang/Documents/projects/api_integrated_llm_experiment/tokenizer_saves/smol_llama-101M-GQA",  # pragma: allowlist secret
        "should_use_autoprocessor": False,
    }
    input = "hello!"
    response = get_response_from_llm_with_tokenizer(
        input=input, model_obj=model_obj, temperature=0.0, max_tokens=100
    )
    assert response is not None
    assert len(response) > 0


@pytest.mark.skip(reason="test to save llm components")
def test_save_models() -> None:
    model_name = "BEE-spoke-data/smol_llama-101M-GQA"

    model_name_short = model_name.split("/")[-1]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

    tokenizer.save_pretrained(
        os.path.join(project_root_path, "tokenizer_saves", model_name_short)
    )
    model.save_pretrained(
        os.path.join(project_root_path, "llm_saves", model_name_short)
    )


@pytest.mark.skip(reason="test to save llm components")
def test_load_models() -> None:
    model_name = "smol_llama-101M-GQA"

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(project_root_path, "tokenizer_saves", model_name)
    )

    model = AutoModelForCausalLM.from_pretrained(
        os.path.join(project_root_path, "llm_saves", model_name), device_map="cpu"
    )

    assert tokenizer is not None
    assert model is not None
