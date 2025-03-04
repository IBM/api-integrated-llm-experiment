from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

from api_integrated_llm.data_models.source_models import (
    EvaluationOutputResponseDataUnit,
)


class ConfusionMatrixMode(str, Enum):
    SET = "set"
    COUNTER = "counter"
    LIST = "list"

    def __str__(self):
        return str(self.value)


class ConfusionMatrixModel(BaseModel):
    true_positive: int = 0
    false_positive: int = 0
    true_negative: int = 0
    false_negative: int = 0
    mode: ConfusionMatrixMode = ConfusionMatrixMode.SET
    num_non_zero_gold: int = 0
    num_is_covered: int = 0

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
    confusion_matrix: Optional[ConfusionMatrixModel] = None

    @staticmethod
    def get_confusion_matrix_metrics_micro(
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
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                confusion_matrix=model.model_copy(deep=True),
            )
        return ConfusionMetrixMetricsModel()


class MicroConfusionMetrixMetricsModel(BaseModel):
    intent_set_metrics: ConfusionMetrixMetricsModel = ConfusionMetrixMetricsModel()
    intent_counter_metrics: ConfusionMetrixMetricsModel = ConfusionMetrixMetricsModel()
    intent_list_metrics: ConfusionMetrixMetricsModel = ConfusionMetrixMetricsModel()
    slot_set_metrics: ConfusionMetrixMetricsModel = ConfusionMetrixMetricsModel()


class ScorerOuputModel(BaseModel):
    confusion_metrix_matrics_micro: MicroConfusionMetrixMetricsModel
    num_examples: int
    percentage_times_full_score: float
    win_rate: Optional[float]
    num_errors_parsing_pred_intent: int
    num_errors_parsing_gold_intent: int
    num_errors_parsing_pred_slot: int
    num_errors_parsing_gold_slot: int
    num_pred_examples_w_parsing_errors: int
    num_gold_examples_w_parsing_errors: int
    error_messages: List[str]
    parsing_error_messages: List[str]
    model_temperature: int
    model_max_tokens: int
    evaluation_source: List[EvaluationOutputResponseDataUnit]
    gold_output_intent: List[List[Union[str, Dict[str, Any]]]]
    pred_output_intent: List[List[Union[str, Dict[str, Any]]]]
    gold_output_slot: List[List[Union[str, Dict[str, Any]]]]
    pred_output_slot: List[List[Union[str, Dict[str, Any]]]]
