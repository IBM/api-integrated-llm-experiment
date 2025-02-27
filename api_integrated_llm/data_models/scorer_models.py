from __future__ import annotations
from enum import Enum
from typing import Optional
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

    def is_valid_model(self) -> bool:
        return (
            self.true_positive > 0
            or self.true_negative > 0
            or self.false_negative > 0
            or self.false_positive > 0
        )


class ConfusionMetrixMetricsModel(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None

    @staticmethod
    def get_confusion_matrix_metrics(
        model: ConfusionMatrixModel,
    ) -> ConfusionMetrixMetricsModel:
        if model.is_valid_model():
            accuracy = (model.true_positive + model.true_negative) / (
                model.true_positive
                + model.true_negative
                + model.false_positive
                + model.false_negative
            )
            precision = (
                (model.true_positive / (model.true_positive + model.false_positive))
                if model.true_positive + model.false_positive > 0
                else None
            )
            recall = (
                (model.true_positive / (model.true_positive + model.false_negative))
                if model.true_positive + model.false_negative > 0
                else None
            )
            f1: Optional[float] = None
            if precision is not None and recall is not None:
                f1 = (
                    (2 * precision * recall / (precision + recall))
                    if precision + recall > 0
                    else None
                )

            return ConfusionMetrixMetricsModel(
                accuracy=accuracy, precision=precision, recall=recall, f1=f1
            )
        return ConfusionMetrixMetricsModel()
