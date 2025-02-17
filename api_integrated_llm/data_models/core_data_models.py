from typing import Any, Dict, List
from pydantic import BaseModel


class DataUnit(BaseModel):
    input: str
    output: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    gold_answer: Any
