from datetime import datetime
import hashlib
import os
from pathlib import Path
import pickle
import pprint
from typing import Any, Dict, Optional
import json
from typing import List
import uuid
from pydantic import BaseModel


def get_uuid4_str() -> str:
    return uuid.uuid4().hex


def create_folders_recirsively_if_not_exist(tmp_path: Path) -> None:
    base_path = os.path.basename(os.path.normpath(tmp_path))
    directory_path = (
        os.path.dirname(os.path.abspath(tmp_path))  # file path
        if "." in base_path
        else os.path.abspath(tmp_path)  # folder path
    )

    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)


def write_pickle_file(file_path: Path, obj: Any) -> None:
    create_folders_recirsively_if_not_exist(tmp_path=file_path)

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)


def read_pickle_file(file_path: Path) -> Any:
    data = None
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def get_dict_from_json(file_path: Path) -> Dict[str, Any]:
    tmp_dict = {}
    with open(file_path, "r") as f:
        tmp_dict = json.load(f)
    return tmp_dict


def write_txt_file(file_path: Path, text: str) -> None:
    create_folders_recirsively_if_not_exist(tmp_path=file_path)

    with open(file_path, "w") as f:
        f.write(text)


def write_json_from_dict(file_path: Path, dic: Dict) -> None:
    create_folders_recirsively_if_not_exist(tmp_path=file_path)

    with open(file_path, "w") as outfile:
        json.dump(dic, outfile)


def write_json(file_path: Path, base_model: BaseModel) -> None:
    create_folders_recirsively_if_not_exist(tmp_path=file_path)

    with open(file_path, "w") as f:
        f.write(base_model.model_dump_json(indent=2))


def write_jsonl(file_path: Path, jsons: List[BaseModel]) -> None:
    create_folders_recirsively_if_not_exist(tmp_path=file_path)

    with open(file_path, "w") as f:
        for item in jsons:
            f.write(item.model_dump_json() + "\n")


def write_list_dict_jsonl(file_path: Path, dicts: List[Dict[Any, Any]]) -> None:
    create_folders_recirsively_if_not_exist(tmp_path=file_path)

    with open(file_path, "w") as f:
        for item in dicts:
            f.write(json.dumps(item) + "\n")


def get_base_models_from_jsonl(
    file_path: Path, base_model: BaseModel
) -> List[BaseModel]:
    outputs: List[BaseModel] = list()
    with open(file_path, "r") as f:
        json_list = list(f)

    for json_str in json_list:
        tmp_dict = json.loads(json_str)
        try:
            model = base_model.model_validate(tmp_dict)
            outputs.append(model)
        except Exception as e:
            print(e)
    return outputs


def get_base_models_from_json_list(
    file_path: Path, base_model: BaseModel
) -> List[BaseModel]:
    raw_list = []
    outputs: List[BaseModel] = list()
    with open(file_path, "r") as f:
        raw_list = json.load(f)

    for json_dict in raw_list:
        try:
            model = base_model.model_validate(json_dict)
            outputs.append(model)
        except Exception as e:
            print(e)
    return outputs


def get_base_model_from_json(file_path: Path, base_model: BaseModel) -> BaseModel:
    with open(file_path, "r") as f:
        tmp_dict = json.load(f)

    try:
        new_model = base_model.model_validate(tmp_dict)
    except Exception as e:
        print(e)
        raise Exception(f"Model Parsing failed: {file_path}")

    return new_model


def get_models_from_jsonl(file_path: Path, model: BaseModel) -> List[BaseModel]:
    outputs: List[BaseModel] = list()
    with open(file_path, "r") as f:
        json_list = list(f)

    for json_str in json_list:
        tmp_dict = json.loads(json_str)
        outputs.append(model.model_validate(tmp_dict))
    return outputs


def get_list_dict_from_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = list()
    with open(file_path, "r") as f:
        json_list = list(f)

    for json_str in json_list:
        tmp_dict = json.loads(json_str)
        outputs.append(tmp_dict)
    return outputs


def get_date_time_str() -> str:
    now = datetime.now()  # current date and time
    return now.strftime("%m_%d_%Y_%H_%M_%S")


def get_file_path(
    file_path_without_extension: str, key_words: List[str], extension: str
) -> str:
    return (
        file_path_without_extension
        + "_"
        + "_".join(key_words)
        + "_"
        + get_date_time_str()
        + "."
        + extension
    )


def get_uuid_str() -> str:
    return str(uuid.uuid4())


def write_json_with_name_parts(
    name_parts: List[str],
    extension: str,
    output_folder_path: Path,
    base_model: BaseModel,
) -> str:
    """
    returns error messages
    """
    error_messages = ""
    file_name = "_".join(name_parts) + extension
    file_path = os.path.join(output_folder_path, file_name)
    try:
        write_json(file_path=Path(file_path), base_model=base_model)
    except Exception as e:
        error_messages = str(e)
        print(error_messages)

    return error_messages


def check_paths(paths: List[Path]) -> None:
    error_messages: List[str] = []
    for tmp_path in paths:
        if not os.path.exists(tmp_path):
            error_messages.append(f"{tmp_path} does not exist.")
    if len(error_messages) > 0:
        pprint.pp(error_messages)
        raise Exception("Invalid Directory Path")


def print_return_base_model(model: BaseModel) -> BaseModel:
    print()
    pprint.pp(model)

    return model


def get_files_in_folder(
    folder_path: Path, file_extension: Optional[str] = None
) -> List[Path]:
    return (
        [
            Path(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(folder_path)
            for f in filenames
            if os.path.splitext(f)[1] == ("." + file_extension)
        ]
        if file_extension is not None
        else [
            Path(os.path.join(dp, f))
            for dp, dn, filenames in os.walk(folder_path)
            for f in filenames
        ]
    )


def get_base_models_from_folder(
    folder_path: Path,
    file_extension: str,
    base_model: BaseModel,
) -> List[BaseModel]:
    json_file_paths = get_files_in_folder(
        folder_path=folder_path,
        file_extension=file_extension,
    )
    models: List[BaseModel] = []

    for file_path in json_file_paths:
        try:
            gym_source_raw = get_base_model_from_json(
                file_path=Path(file_path), base_model=base_model
            )
            models.append(gym_source_raw)
        except Exception as e:
            print(e)

    return models


def get_base_model_from_llm_generated_text(
    text: str, base_model: BaseModel
) -> BaseModel:
    instance = base_model()
    try:
        start_idx = text.index("{")
        end_idx = text.rfind("}")
        instance = base_model.model_validate_json(
            text[start_idx : (end_idx + 1)]  # noqa: E203
        )
    except Exception as e:
        print(e)

    return instance


def get_file_content_str_from_file_names(
    folder_absolute_path: Path, file_names: List[str]
) -> List[str]:
    contents: List[str] = []
    for file_name in file_names:
        tmp_path = os.path.join(folder_absolute_path, file_name)
        with open(tmp_path, "r") as f:
            contents.append(f.read())
    return contents


def get_file_content_dict_from_file_names(
    folder_absolute_path: Path, file_names: List[str]
) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []
    for file_name in file_names:
        tmp_path = os.path.join(folder_absolute_path, file_name)
        try:
            with open(tmp_path, "r") as f:
                contents.append(json.load(f))
        except Exception as e:
            print(e)
    return contents


def get_hash_str_from_dict(tmp_dict: Dict[str, Any]) -> str:
    dhash = hashlib.md5()
    encoded = json.dumps(tmp_dict, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_dataset_name(file_path: Path) -> str:
    json_dict = get_dict_from_json(file_path)
    data = json_dict["data"]
    return data[0]["dataset_name"][:]


def get_file_name_without_extension(file_path: Path) -> str:
    file_name = str(file_path).split("/")[-1]
    return "".join(file_name.split(".")[:-1])


def get_dataset_name_from_file_path(file_path: Path) -> str:
    file_name = str(file_path).split("/")[-1]
    return file_name.replace(".jsonl", "")


def get_json_dict_from_txt(txt: str) -> Dict[str, Any]:
    start_idx = txt.index("{")
    end_idx = txt.rfind("}")

    if start_idx >= end_idx:
        raise Exception("text does not contain json string")

    truncated_json_str = txt[start_idx : (end_idx + 1)]  # noqa: E203
    json_dict = {}
    try:
        json_dict = json.loads(truncated_json_str)
    except Exception as e:
        print(e)
        raise Exception("text does not contain a valid json string")

    return json_dict
