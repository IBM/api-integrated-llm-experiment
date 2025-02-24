from pydantic import BaseModel


class CommonErrorModel(BaseModel):
    error: str
    file: str
