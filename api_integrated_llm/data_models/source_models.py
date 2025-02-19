from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

from api_integrated_llm.helpers.file_helper import get_uuid4_str


class PropertyItem(BaseModel):
    description: Optional[str] = None
    type: Optional[str] = None


class ParametersModel(BaseModel):
    properties: Optional[Dict[str, PropertyItem]] = None
    required: Optional[List[str]] = None
    type: Optional[str] = None


class ToolDescriptionModel(BaseModel):
    description: Optional[str] = None
    name: Optional[str] = None
    parameters: ParametersModel = ParametersModel()
    output_parameters: ParametersModel = ParametersModel()


class ToolItemModel(BaseModel):
    description: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = dict()


class QueryItemDataModel(BaseModel):
    name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = dict()
    label: Optional[str] = None


class QueryKeyValueDescriptionDataModel(BaseModel):
    key_name: Optional[str] = None
    description: Optional[str] = None
    dtype: Optional[str] = None


class DataUnit(BaseModel):
    input: Optional[str] = None
    output: Optional[List[QueryItemDataModel]] = None
    gold_answer: Optional[Union[List[Any], str, int, float]] = None
    tools: Optional[List[ToolItemModel]] = list()


class QuerySourceDataModel(BaseModel):
    sample_id: str = get_uuid4_str()
    input: Optional[str] = None
    output: Optional[List[QueryItemDataModel]] = None
    gold_answer: Optional[Union[List[Any], str, int, float]] = None
    original_output: Optional[List[QueryItemDataModel]] = None
    initialization_step: Optional[QueryItemDataModel] = None
    tools: Optional[List[ToolItemModel]] = list()
    key_values_and_descriptions: Optional[
        List[QueryKeyValueDescriptionDataModel]
    ] = None

    def get_data_unit_model(self) -> DataUnit:
        return DataUnit(
            input=self.input,
            output=self.output,
            gold_answer=self.gold_answer,
            tools=self.tools,
        )


class QuerySourceModel(BaseModel):
    data: Optional[List[QuerySourceDataModel]] = None
    global_api_pool: Optional[Dict[str, ToolDescriptionModel]] = None
    win_rate: Optional[float] = None


class ExampleDataModel(BaseModel):
    data: List[DataUnit] = list()


class EvaluationOutputDataUnit(BaseModel):
    sample_id: str
    input: str
    output: Optional[List[QueryItemDataModel]] = None
    gold_answer: Optional[Union[List[Any], str, int, float]] = None


class EvaluationOutputResponseDataUnit(EvaluationOutputDataUnit):
    generated_text: str = ""
    llm_model_id: str = ""
    source_file_path: Path = Path(__file__)
    dataset_name: str = ""
    temperature: float = -1.0
    max_tokens: int = 1500

    @staticmethod
    def get_model_from_output_unit(
        model: EvaluationOutputDataUnit,
    ) -> EvaluationOutputResponseDataUnit:
        return EvaluationOutputResponseDataUnit(
            sample_id=model.sample_id,
            input=model.input,
            output=model.output,
            gold_answer=model.gold_answer,
        )
