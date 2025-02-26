from datetime import timedelta
from typing import Any
from pydantic import BaseModel


class CommonErrorModel(BaseModel):
    error: str
    file: str
    payload: Any


class HttpResponseModel(BaseModel):
    elapsed: timedelta
    response_txt: str
    status: int
    error_message: str = ""
