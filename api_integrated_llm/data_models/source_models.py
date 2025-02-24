from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel

from api_integrated_llm.helpers.file_helper import get_uuid4_str


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


class QuerySourceModel(BaseModel):
    data: Optional[List[QuerySourceDataModel]] = None
    global_api_pool: Optional[Dict[str, Any]] = None
    win_rate: Optional[float] = None


class ExampleDataModel(BaseModel):
    data: List[DataUnit] = list()


class EvaluationOutputDataUnit(BaseModel):
    sample_id: Union[str, int]
    input: str
    output: Optional[Union[List[QueryItemDataModel], str]] = None
    gold_answer: Optional[Union[List[Any], str, int, float]] = None


class EvaluationOutputResponseDataUnit(EvaluationOutputDataUnit):
    generated_text: str = ""
    llm_model_id: str = ""
    source_file_path: str = str(Path(__file__))
    dataset_name: str = ""
    temperature: float = -1.0
    max_tokens: int = 1500

    @staticmethod
    def get_model_from_output_unit(
        data_model: EvaluationOutputDataUnit,
    ) -> EvaluationOutputResponseDataUnit:
        return EvaluationOutputResponseDataUnit(
            sample_id=data_model.sample_id,
            input=data_model.input,
            output=data_model.output,
            gold_answer=data_model.gold_answer,
        )

    def get_basic_strs(self) -> Tuple[str, str, str, str]:
        return (
            ("temperature_" + str(self.temperature).replace(".", "_")),
            ("maxtokens_" + str(self.max_tokens)),
            self.dataset_name[:],
            self.llm_model_id[:],
        )


class ScorerOuputModel(BaseModel):
    p_intent: float
    r_intent: float
    f1_intent: float
    p_slot: Optional[float]
    r_slot: Optional[float]
    f1_slot: Optional[float]
    num_examples: int
    accuracy_combined: float
    percentage_times_full_score: float
    win_rate: Optional[float]
    num_errors_parsing_pred_intent: int
    num_errors_parsing_gold_intent: int
    num_errors_parsing_pred_slot: int
    num_errors_parsing_gold_slot: int
    num_pred_examples_w_parsing_errors: int
    error_messages: List[str]
    model_temperature: int
    model_max_tokens: int
    evaluation_source: List[EvaluationOutputResponseDataUnit]
    gold_output_intent: List[Any]
    pred_output_intent: List[Any]
    gold_output_slot: List[Any]
    pred_output_slot: List[Any]
