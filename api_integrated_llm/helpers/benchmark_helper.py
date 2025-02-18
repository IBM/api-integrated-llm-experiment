from copy import deepcopy
from typing import Dict


def get_model_id_obj_dict() -> Dict[str, Dict[str, str]]:
    model_ids_set = set(
        [
            "granite-3.1-8b-instruct",
            "Llama-3.1-8B-Instruct",
            "mixtral_8x7b_instruct_v01",
            "llama-3-1-70b-instruct",
            "llama-3-1-405b-instruct-fp8",
            "DeepSeek-V3",
            "Mixtral-8x22B-Instruct-v0.1",
            "llama-3-3-70b-instruct",
        ]
    )
    model_id_obj_dict = dict()

    model_info_dict = {
        "DeepSeek-V3": {
            "inference_type": "RITS",
            "model": "deepseek-ai/DeepSeek-V3",
            "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/deepseek-v3/v1",
        },
        "Granite-20B-FunctionCalling": {
            "inference_type": "RITS",
            "model": "ibm-granite/granite-20b-code-instruct-unified-api",
            "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-20b-code-instruct-uapi/v1",
        },
        "Mixtral-8x22B-Instruct-v0.1": {
            "inference_type": "RITS",
            "model": "mistralai/mixtral-8x22B-instruct-v0.1",
            "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x22b-instruct-v01/v1",
        },
        "mixtral_8x7b_instruct_v01": {
            "inference_type": "RITS",
            "model": "mistralai/mixtral-8x7B-instruct-v0.1",
            "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/mixtral-8x7b-instruct-v01/v1",
        },
        "granite-3.1-8b-instruct": {
            "inference_type": "RITS",
            "model": "ibm-granite/granite-3.1-8b-instruct",
            "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/granite-3-1-8b-instruct/v1",
        },
        "Llama-3.1-8B-Instruct": {
            "inference_type": "RITS",
            "model": "meta-llama/Llama-3.1-8B-Instruct",
            "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-1-8b-instruct/v1",
        },
        "llama-3-1-70b-instruct": {
            "inference_type": "RITS",
            "model": "meta-llama/llama-3-1-70b-instruct",
            "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-1-70b-instruct/v1",
        },
        "llama-3-1-405b-instruct-fp8": {
            "inference_type": "RITS",
            "model": "meta-llama/llama-3-1-405b-instruct-fp8",
            "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-1-405b-instruct-fp8/v1",
        },
        "Llama-3.2-11B-Vision-Instruct": {
            "inference_type": "RITS",
            "model": "meta-llama/Llama-3.2-11B-Vision-Instruct",
            "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-2-11b-instruct/v1",
        },
        "Llama-3.2-90B-Vision-Instruct": {
            "inference_type": "RITS",
            "model": "meta-llama/Llama-3.2-90B-Vision-Instruct",
            "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-2-90b-instruct/v1",
        },
        "llama-3-3-70b-instruct": {
            "inference_type": "RITS",
            "model": "meta-llama/llama-3-3-70b-instruct",
            "endpoint": "https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com/llama-3-3-70b-instruct/v1",
        },
    }

    for model_id in model_info_dict.keys():
        if model_id in model_ids_set:
            model_id_obj_dict[model_id] = deepcopy(model_info_dict[model_id])

    return model_id_obj_dict
