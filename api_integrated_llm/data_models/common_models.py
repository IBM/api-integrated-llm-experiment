from datetime import timedelta
from pydantic import BaseModel


class CommonErrorModel(BaseModel):
    error: str
    file: str


class HttpResponseModel(BaseModel):
    elapsed: timedelta
    response_txt: str
    status: int
    error_message: str = ""
