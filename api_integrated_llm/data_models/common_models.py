from datetime import timedelta
from typing import Optional
from pydantic import BaseModel


class CommonErrorModel(BaseModel):
    error: str
    file: Optional[str] = None
    payload: Optional[str] = None


class HttpResponseModel(BaseModel):
    elapsed: timedelta
    response_txt: str
    status: int
    error_message: str = ""
