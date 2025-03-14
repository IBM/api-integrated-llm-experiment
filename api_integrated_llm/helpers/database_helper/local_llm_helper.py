from typing import Dict, List, Optional, Tuple
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    GenerationConfig,
)

from api_integrated_llm.data_models.source_models import EvaluationOutputDataUnit

LLM_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizers_dict = {}
models_dict = {}
processors_dict = {}


def get_response_from_llm_with_tokenizer(
    input: str, model_obj: Dict[str, str], temperature: float, max_tokens: int
) -> Optional[str]:
    if "tokenizer" not in model_obj or "model" not in model_obj:
        return None

    if model_obj["tokenizer"] not in tokenizers_dict:
        tokenizers_dict[model_obj["tokenizer"]] = AutoTokenizer.from_pretrained(
            model_obj["tokenizer_url"]
            if "tokenizer_url" in model_obj
            else model_obj["tokenizer"]
        )

    if model_obj["model"] not in models_dict:
        models_dict[model_obj["model"]] = AutoModelForCausalLM.from_pretrained(
            (
                model_obj["model_url"]
                if "model_url" in model_obj
                else model_obj["model"]
            ),
            torch_dtype=torch.bfloat16 if "cuda" in LLM_DEVICE else torch.float16,
            device_map="auto",
        )

    tokenizer = tokenizers_dict[model_obj["tokenizer"]]

    if max_tokens > tokenizer.model_max_length:
        max_tokens = tokenizer.model_max_length - 1

    model = models_dict[model_obj["model"]]

    input_tokens_dict = tokenizer(input, return_tensors="pt")
    input_tokens_dict = {k: v.to(model.device) for k, v in input_tokens_dict.items()}
    output = model.generate(
        **input_tokens_dict, do_sample=False, max_new_tokens=max_tokens
    )
    response = tokenizer.decode(
        output[0][len(input_tokens_dict["input_ids"][0]) :],  # noqa: E203
        skip_special_tokens=True,
    )

    return response


def get_response_from_llm_with_autoprocessor(
    input: str, model_obj: Dict[str, str], temperature: float, max_tokens: int
) -> Optional[str]:
    """
    This function supports only cuda
    """
    if "model" not in model_obj:
        return None

    if model_obj["model"] not in processors_dict:
        processors_dict[model_obj["model"]] = AutoProcessor.from_pretrained(
            model_obj["model"], trust_remote_code=True
        )

    if model_obj["model"] not in models_dict:
        models_dict[model_obj["model"]] = AutoModelForCausalLM.from_pretrained(
            model_obj["model"],
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation="flash_attention_2",
        ).cuda()

    processor = processors_dict[model_obj["model"]]
    model = models_dict[model_obj["model"]]

    # Load generation config
    generation_config = GenerationConfig.from_pretrained(model_obj["model"])
    inputs = processor(text=input, return_tensors="pt").to("cuda:0")
    generate_ids = model.generate(
        **inputs,
        temperature=temperature,
        max_new_tokens=max_tokens,
        generation_config=generation_config,
    )
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]  # noqa: E203
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return response


def get_responses_from_local_llm(
    test_data: List[EvaluationOutputDataUnit],
    model_obj: Dict[str, str],
    temperature: float,
    max_tokens: int,
) -> Tuple[List[List[str]], List[str]]:
    error_messages: List[str] = []
    responses: List[str] = []
    for sample in test_data:
        try:
            response = (
                get_response_from_llm_with_autoprocessor(
                    input=sample.input,
                    model_obj=model_obj,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if (
                    "should_use_autoprocessor" in model_obj
                    and model_obj["should_use_autoprocessor"]
                )
                else get_response_from_llm_with_tokenizer(
                    input=sample.input,
                    model_obj=model_obj,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )
            responses.append(response if response is not None else "")
        except Exception as e:
            print(e)
            model_name = model_obj["model"] if "model" in model_obj else "default_model"
            error_message = str(e)
            error_messages.append(
                f"Error message from local LLM {model_name}: {error_message}"
            )

    return responses, error_messages
