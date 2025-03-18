from __future__ import annotations
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel

from api_integrated_llm.helpers.file_helper import get_hash, get_uuid4_str


class PropertyItem(BaseModel):
    description: Optional[str] = None
    type: Optional[str] = None


class ParametersModel(BaseModel):
    properties: Optional[Union[Dict[str, Union[PropertyItem, str]], str]] = None
    required: Optional[List[str]] = None
    type: Optional[str] = None


class ToolDescriptionModel(BaseModel):
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: Union[Dict[str, Any], ParametersModel] = ParametersModel()
    output_parameters: Union[Dict[str, Any], ParametersModel] = ParametersModel()


class ToolItemModel(BaseModel):
    description: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[Union[Dict[str, Any], str]] = dict()


class QueryItemDataModel(BaseModel):
    name: Optional[str] = None
    arguments: Optional[Union[Dict[str, Any], str]] = dict()
    label: Optional[str] = None


class QueryKeyValueDescriptionDataModel(BaseModel):
    key_name: Optional[Union[str, float, int]] = None
    description: Optional[Union[str, float, int]] = None
    dtype: Optional[Union[str, float, int]] = None


class DataUnit(BaseModel):
    input: Optional[str] = None
    output: Optional[List[QueryItemDataModel]] = None
    gold_answer: Optional[Union[List[Any], str, int, float]] = None
    tools: Optional[List[ToolItemModel]] = list()


class QuerySourceDataModel(BaseModel):
    sample_id: Optional[Union[str, int]] = get_uuid4_str()
    input: Optional[str] = None
    output: Optional[List[QueryItemDataModel]] = None
    gold_answer: Optional[Union[List[Any], str, int, float]] = None
    original_output: Optional[List[Any]] = None
    initialization_step: Optional[Any] = None
    tools: Optional[List[ToolItemModel]] = list()
    key_values_and_descriptions: Optional[
        List[QueryKeyValueDescriptionDataModel]
    ] = None
    ignore: Optional[bool] = False

    def get_data_unit_model(self) -> DataUnit:
        return DataUnit(
            input=self.input,
            output=self.output,
            gold_answer=self.gold_answer,
            tools=self.tools,
        )

    def get_hash(self) -> str:
        return get_hash(self.model_dump_json())

    def get_tools_raw(self) -> List[Dict[str, Any]]:
        if self.tools is None:
            return []

        return [
            # {"type": "function", "function": tool.model_dump()} for tool in self.tools
            tool.model_dump()
            for tool in self.tools
        ]


class QuerySourceModel(BaseModel):
    data: Optional[List[QuerySourceDataModel]] = None
    global_api_pool: Optional[Dict[str, Any]] = None
    win_rate: Optional[float] = None
    dataset: Optional[str] = None


class ExampleDataModel(BaseModel):
    data: List[DataUnit] = list()


class ConversationRoleModel(str, Enum):
    SYSTEM = "system"
    USER = "user"

    def __str__(self):
        return str(self.value)


class ConversationUnit(BaseModel):
    role: ConversationRoleModel = ConversationRoleModel.USER
    content: str = ""


class EvaluationOutputDataUnit(BaseModel):
    sample_id: Union[str, int]
    input: str
    output: Optional[Union[List[QueryItemDataModel], str]] = None
    gold_answer: Optional[Union[List[Any], str, int, float]] = None
    messages: Optional[List[ConversationUnit]] = None
    tools: Optional[List[Dict[str, Any]]] = None

    def get_message_raw(self) -> List[Dict[str, Any]]:
        if self.messages is None:
            return []
        return [obj.model_dump() for obj in self.messages]

    def get_tools_str(self) -> str:
        if self.tools is None:
            return ""
        return json.dumps(self.tools)


class EvaluationOutputResponseDataUnit(EvaluationOutputDataUnit):
    generated_text: str = ""
    llm_model_id: str = ""
    source_file_path: str = str(Path(__file__))
    dataset_name: str = ""
    temperature: float = -1.0
    max_tokens: int = 1500
    is_agent: bool = False
    predicted_function_calls: List[str] = []
    gold_function_calls: List[str] = []
    num_preciedtion_parsing_errors: Optional[int] = None

    @staticmethod
    def get_model_from_output_unit(
        data_model: EvaluationOutputDataUnit,
    ) -> EvaluationOutputResponseDataUnit:
        return EvaluationOutputResponseDataUnit(
            sample_id=data_model.sample_id,
            input=(
                data_model.input
                if (data_model.messages is None or len(data_model.messages) == 0)
                else json.dumps(
                    [message.model_dump() for message in data_model.messages]
                )
            ),
            output=data_model.output,
            gold_answer=data_model.gold_answer,
        )

    def get_basic_strs(self) -> Tuple[str, str, str, str, str]:
        return (
            ("temperature_" + str(self.temperature).replace(".", "_")),
            ("maxtokens_" + str(self.max_tokens)),
            self.dataset_name,
            self.llm_model_id.split("/")[-1],
            ("agent" if self.is_agent else "llm"),
        )

    def get_dataset_basic_info(
        self,
    ) -> Tuple[str, str, Path, float, int, str]:
        return (
            self.llm_model_id.split("/")[-1],
            self.dataset_name,
            Path(self.source_file_path),
            self.temperature,
            self.max_tokens,
            ("agent" if self.is_agent else "llm"),
        )
