from enum import Enum

from pydantic import BaseModel


class ConfusionMatrixMode(str, Enum):
    SET = "set"
    MULTISET = "multiset"
    LIST = "list"

    def __str__(self):
        return str(self.value)


class ConfusionMatrixModel(BaseModel):
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    mode: ConfusionMatrixMode = ConfusionMatrixMode.SET
    is_non_zero_gold: bool = False
    is_covered: bool = False
