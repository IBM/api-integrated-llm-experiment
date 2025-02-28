from collections import OrderedDict
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
) -> HttpResponseModel:
    async with aiohttp.ClientSession() as session:
        start_time = time.monotonic()
        if headers is not None:
            async with session.post(
                url, json=obj, headers=headers, timeout=timeout
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


async def get_response_from_post_request_async(
    obj: Dict[str, Any],
    url: str,
    headers: Optional[Dict[str, Any]] = None,
    timeout: int = 600,
) -> Tuple[Optional[Union[str, List[str]]], str, float]:
    text_response: Optional[Union[str, List[str]]] = None
    error_message: str = ""
    lag: Optional[timedelta] = None
    try:
        response_model = await fetch_data_post(
            url=url,
            obj=obj,
            headers=headers,
            timeout=timeout,
        )
        lag = response_model.elapsed
        payload = json.loads(response_model.response_txt, object_pairs_hook=OrderedDict)

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
        payload = json.loads(response.text, object_pairs_hook=OrderedDict)

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


async def get_response_from_RITS_async(
    id_model: str,
    model_resource: str,
    api_key: str,
    contents: List[str],
    max_tokens: int,
    temperature: float = 0.0,
    n: int = 1,
    timeout: int = 240,
) -> Tuple[Optional[Union[str, List[str]]], str, float]:
    return await get_response_from_post_request_async(
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


async def generate_rits_response_async(
    prompt: str,
    temperature: float,
    max_tokens: int,
    model_name: str,
    model_resource: str,
):
    try:
        resp = await get_response_from_RITS_async(
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


async def get_responses_from_async(
    test_data: List[EvaluationOutputDataUnit],
    model_obj: Dict[str, str],
    temperature: float,
    max_tokens: int,
) -> List[str]:
    tasks = [
        generate_rits_response_async(
            prompt=sample.input[:],
            temperature=temperature,
            max_tokens=max_tokens,
            model_name=model_obj["model"][:],
            model_resource=model_obj["endpoint"].split("/")[-2][:],
        )
        for sample in test_data
    ]

    return await asyncio.gather(*tasks)
