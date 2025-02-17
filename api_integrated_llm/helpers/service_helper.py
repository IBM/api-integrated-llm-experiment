from copy import deepcopy
from datetime import timedelta
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from multiprocessing import Pool
import requests


def get_response_from_post_request(
    obj: Dict[str, Any],
    url: str,
    headers: Optional[Dict[str, Any]] = None,
    timeout: int = 240,
) -> Tuple[Optional[Union[str, List[str]]], str, float]:
    text_response: Optional[Union[str, List[str]]] = None
    error_message: str = ""
    lag: Optional[timedelta] = None
    try:
        response = (
            requests.post(url, json=obj, timeout=timeout)
            if headers is None
            else requests.post(url, json=obj, headers=headers, timeout=timeout)
        )
        lag = response.elapsed
        payload = json.loads(response.text)

        if "response" in payload:  # single response from ollama
            text_response = payload["response"]
        elif "choices" in payload:  # multiple responses from RITS
            text_response = []
            for choice in payload["choices"]:
                if "text" in choice:
                    text_response.append(choice["text"])
    except Exception as e:
        error_message = str(e)
        print(error_message)

    return (
        text_response,
        error_message,
        (-1.0 if lag is None else lag.total_seconds()),
    )


def get_openai_payload(
    prompts: List[str],
    id_model: str,
    temperature: float = 0.0,
    n: int = 1,
    max_tokens: int = 3000,
    seed: int = 123456,
) -> Dict[str, Any]:
    return {
        "model": id_model,
        "prompt": prompts,
        "best_of": 0,
        "echo": False,
        "frequency_penalty": 0,
        "max_tokens": max_tokens,
        "n": n,
        "presence_penalty": 0,
        "seed": seed,
        "stop": "string",
        "stream": False,
        "temperature": temperature,
        "top_p": 1,
        "user": "string",
        "use_beam_search": False,
        "top_k": -1,
        "min_p": 0,
        "repetition_penalty": 1,
        "length_penalty": 1,
        "stop_token_ids": [0],
        "include_stop_str_in_output": False,
        "ignore_eos": False,
        "min_tokens": 0,
        "skip_special_tokens": True,
        "spaces_between_special_tokens": True,
        "add_special_tokens": True,
        "response_format": {
            "type": "text",
            "json_schema": {
                "name": "string",
                "description": "string",
                "schema": {},
                "strict": True,
            },
        },
    }


RITS_BASE_URL = (
    "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com"
)
RITS_COMPLETION_RESOURCE = "v1/completions"


def get_openai_api_headers(api_key: str) -> Dict[str, Any]:
    return {
        "accept": "application/json",
        "RITS_API_KEY": api_key,
        "Content-Type": "application/json",
    }


def get_RITS_model_url(model_resource: str) -> str:
    url_elements = [RITS_BASE_URL, model_resource, RITS_COMPLETION_RESOURCE]
    return "/".join(url_elements)


def get_response_from_RITS(
    id_model: str,
    model_resource: str,
    api_key: str,
    contents: List[str],
    max_tokens: int,
    temperature: float = 0.0,
    n: int = 1,
    timeout: int = 240,
) -> Tuple[Optional[Union[str, List[str]]], str, float]:
    return get_response_from_post_request(
        obj=get_openai_payload(
            prompts=deepcopy(contents),
            id_model=id_model[:],
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            seed=123456,
        ),
        url=get_RITS_model_url(model_resource=model_resource[:]),
        headers=get_openai_api_headers(api_key=api_key[:]),
        timeout=timeout,
    )


def generate_rits_response(prompt, temperature, max_tokens, model_name, model_resource):
    try:
        resp = get_response_from_RITS(
            id_model=model_name[:],
            model_resource=model_resource[:],
            api_key=os.environ["RITS_API_KEY"],
            contents=[prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            n=1,
            timeout=1500,
        )

        return resp[0]
    except Exception as exception:
        return dict(error=exception)


def get_responses_from_pool(
    test_data: List[Dict[str, Any]],
    model_obj: Dict[str, str],
    temperature: float,
    max_tokens: int,
) -> List[str]:
    prompts = []
    responses = []

    for sample in test_data:
        prompts.append(
            (
                sample["input"],
                temperature,
                max_tokens,
                model_obj["model"],
                model_obj["endpoint"].split("/")[-2],
            )
        )

    with Pool(processes=40) as pool:
        responses = pool.starmap(generate_rits_response, prompts)
    return responses
