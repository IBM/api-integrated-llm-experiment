from typing import Any, Dict, List, Optional
from pydantic import BaseModel


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


class QuerySourceModel(BaseModel):
    data: Optional[List[Any]] = None
    global_api_pool: Optional[Dict[str, ToolDescriptionModel]] = None
    win_rate: Optional[float] = None
