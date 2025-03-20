from copy import deepcopy
from datetime import timedelta
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import requests

from api_integrated_llm.data_models.common_models import HttpResponseModel
from api_integrated_llm.data_models.source_models import EvaluationOutputDataUnit

import aiohttp
import asyncio


async def fetch_data_post(
    url: str,
    obj: Dict[str, Any],
    headers: Optional[Dict[str, Any]] = None,
    timeout: int = 240,
    params: Optional[Dict[str, Any]] = None,
) -> HttpResponseModel:
    async with aiohttp.ClientSession() as session:
        start_time = time.monotonic()
        if headers is not None:
            async with session.post(
                url,
                json=obj,
                headers=headers,
                timeout=timeout,
                params=(params if params is not None else {}),
            ) as response:
                return HttpResponseModel(
                    elapsed=timedelta(time.monotonic() - start_time),
                    status=response.status,
                    response_txt=await response.text(),
                )
        else:
            async with session.post(url, json=obj, timeout=timeout) as response:
                return HttpResponseModel(
                    elapsed=timedelta(time.monotonic() - start_time),
                    status=response.status,
                    response_txt=await response.text(),
                )


def get_parsed_payload(
    response_model: HttpResponseModel,
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], str]:
    text_responses: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    error_message: str = ""
    try:
        payload = json.loads(response_model.response_txt)
        if "error" in payload:
            error_message = json.dumps(payload)
            text_responses = error_message
            error_message = json.dumps(payload)

        if "choices" in payload:  # multiple responses from RITS
            for choice in payload["choices"]:
                if "message" in choice:
                    tool_calls = (
                        deepcopy(choice["message"]["tool_calls"])
                        if (
                            "tool_calls" in choice["message"]
                            and choice["message"]["tool_calls"] is not None
                        )
                        else None
                    )
                    text_responses = (
                        choice["message"]["content"]
                        if "content" in choice["message"]
                        else ""
                    )
                elif "text" in choice:
                    text_responses = choice["text"]

                break

        elif "response" in payload:  # single response from ollama
            text_responses = payload["response"]
    except Exception as e:
        error_message = str(e)
        print(f"Error at get_parsed_payload(): {error_message}")

    return text_responses, tool_calls, error_message


async def get_response_from_post_request_async(
    obj: Dict[str, Any],
    url: str,
    headers: Optional[Dict[str, Any]] = None,
    timeout: int = 600,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], str, float]:
    text_responses: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    error_message: str = ""
    lag: Optional[timedelta] = None
    try:
        response_model = await fetch_data_post(
            url=url, obj=obj, headers=headers, timeout=timeout, params=params
        )
        lag = response_model.elapsed
        text_responses, tool_calls, error_message = get_parsed_payload(
            response_model=response_model
        )

        if len(error_message) > 0:
            raise Exception("response model parsing error at get_parsed_payload")

    except Exception as e:
        error_message = str(e)
        print(error_message)

    return (
        text_responses,
        tool_calls,
        error_message,
        (-1.0 if lag is None else lag.total_seconds()),
    )


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


def get_openai_url(url: str, llm_model_id: str) -> str:
    new_url = deepcopy(url)
    new_url = new_url.replace("{MODEL_ID}", llm_model_id)
    return new_url


def get_OPENAI_payload(
    sample: EvaluationOutputDataUnit,
    model_obj: Dict[str, str],
    temperature: float = 0.0,
    max_tokens: int = 3000,
) -> Dict[str, Any]:
    return {
        "model": model_obj.get("model", ""),
        "messages": sample.get_message_raw(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "tools": sample.tools,
        "tool_choice": "auto",
    }


def get_RITZ_payload(
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


def get_RITZ_api_headers(api_key: str) -> Dict[str, Any]:
    return {
        "accept": "application/json",
        "RITS_API_KEY": api_key,
        "Content-Type": "application/json",
    }


def get_OPENAI_api_headers(api_key: str) -> Dict[str, Any]:
    return {
        "accept": "application/json",
        "api-key": api_key,
        "Content-Type": "application/json",
    }


def get_RITS_model_url(model_resource: str) -> str:
    url_elements = [RITS_BASE_URL, model_resource, RITS_COMPLETION_RESOURCE]
    return "/".join(url_elements)


async def get_response_from_RITS_async(
    id_model: str,
    model_resource: str,
    api_key: str,
    contents: List[str],
    max_tokens: int,
    temperature: float = 0.0,
    n: int = 1,
    timeout: int = 240,
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], str, float]:
    return await get_response_from_post_request_async(
        obj=get_RITZ_payload(
            prompts=deepcopy(contents),
            id_model=id_model[:],
            temperature=temperature,
            n=n,
            max_tokens=max_tokens,
            seed=123456,
        ),
        url=get_RITS_model_url(model_resource=model_resource[:]),
        headers=get_RITZ_api_headers(api_key=api_key[:]),
        timeout=timeout,
    )


def get_OPENAI_query_parameters(model_obj: Dict[str, str]) -> Dict[str, Any]:
    return {"api-version": model_obj.get("api-version", "")}


async def get_response_from_OPENAI_async(
    sample: EvaluationOutputDataUnit,
    model_obj: Dict[str, str],
    max_tokens: int,
    temperature: float = 0.0,
    timeout: int = 240,
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]], str, float]:
    return await get_response_from_post_request_async(
        obj=get_OPENAI_payload(
            sample=sample,
            model_obj=model_obj,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        url=get_openai_url(
            url=model_obj.get("endpoint", ""),
            llm_model_id=model_obj.get("model-id", ""),
        ),
        headers=get_OPENAI_api_headers(
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", "")
        ),
        timeout=timeout,
        params=get_OPENAI_query_parameters(model_obj=model_obj),
    )


def handle_txt_from_llm_service(
    payload: Tuple[
        Union[str, list[str], None], Optional[list[dict[str, Any]]], str, float
    ],
) -> Optional[str]:
    if not isinstance(payload, tuple):
        return None
    content = payload[0]
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list) and len(content) > 0:
        return content[0]
    return None


async def generate_llm_response_from_service_async(
    sample: EvaluationOutputDataUnit,
    temperature: float,
    max_tokens: int,
    model_obj: Dict[str, str],
) -> Tuple[Optional[str], Optional[List[Dict[str, Any]]]]:
    prompt = sample.input[:]
    model_name = model_obj["model"][:]
    model_resource = model_obj["endpoint"].split("/")[-2][:]

    try:
        resp = (
            await get_response_from_OPENAI_async(
                sample=sample,
                model_obj=deepcopy(model_obj),
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=1500,
            )
            if (
                ("inference_type" in model_obj)
                and (model_obj["inference_type"] == "OPENAI")
            )
            else await get_response_from_RITS_async(
                id_model=model_name[:],
                model_resource=model_resource[:],
                api_key=os.environ.get("RITS_API_KEY", ""),
                contents=[prompt],
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
                timeout=1500,
            )
        )

        return handle_txt_from_llm_service(payload=resp), resp[1]
    except Exception as e:
        error_message = str(e)
        print(f"Error at generate_llm_response_from_service_async: {error_message}")

    return None, None


async def get_responses_from_async(
    test_data: List[EvaluationOutputDataUnit],
    model_obj: Dict[str, str],
    temperature: float,
    max_tokens: int,
) -> List[Tuple[Optional[str], Optional[List[Dict[str, Any]]]]]:
    tasks = [
        generate_llm_response_from_service_async(
            sample=sample,
            temperature=temperature,
            max_tokens=max_tokens,
            model_obj=deepcopy(model_obj),
        )
        for sample in test_data
    ]

    return await asyncio.gather(*tasks)


def get_responses_from_sync(
    test_data: List[EvaluationOutputDataUnit],
    model_obj: Dict[str, str],
    temperature: float,
    max_tokens: int,
) -> List[Tuple[Optional[str], Optional[List[Dict[str, Any]]]]]:
    responses: List[Tuple[Optional[str], Optional[List[Dict[str, Any]]]]] = []
    for sample in test_data:
        response = asyncio.run(
            generate_llm_response_from_service_async(
                sample=sample,
                temperature=temperature,
                max_tokens=max_tokens,
                model_obj=deepcopy(model_obj),
            )
        )
        responses.append(response)

    return responses
