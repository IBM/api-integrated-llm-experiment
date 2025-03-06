from __future__ import annotations
from enum import Enum
import statistics
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

    def add(self, confusion_matrix: ConfusionMatrixModel) -> None:
        self.true_positive += confusion_matrix.true_positive
        self.true_negative += confusion_matrix.true_negative
        self.false_positive += confusion_matrix.false_positive
        self.false_negative += confusion_matrix.false_negative
        self.num_non_zero_gold += confusion_matrix.num_non_zero_gold
        self.num_is_covered += confusion_matrix.num_is_covered


class ConfusionMetrixMetricsModel(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    confusion_matrix: Optional[ConfusionMatrixModel] = None

    @staticmethod
    def get_confusion_matrix_metrics_micro(
        confusion_matrix: ConfusionMatrixModel,
    ) -> ConfusionMetrixMetricsModel:
        if confusion_matrix.is_valid_model():
            tot_num = (
                confusion_matrix.true_positive
                + confusion_matrix.true_negative
                + confusion_matrix.false_positive
                + confusion_matrix.false_negative
            )
            accuracy = (
                (confusion_matrix.true_positive + confusion_matrix.true_negative)
                / (tot_num)
                if tot_num > 0
                else None
            )
            precision = (
                (
                    confusion_matrix.true_positive
                    / (confusion_matrix.true_positive + confusion_matrix.false_positive)
                )
                if confusion_matrix.true_positive + confusion_matrix.false_positive > 0
                else None
            )
            recall = (
                (
                    confusion_matrix.true_positive
                    / (confusion_matrix.true_positive + confusion_matrix.false_negative)
                )
                if confusion_matrix.true_positive + confusion_matrix.false_negative > 0
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
                confusion_matrix=confusion_matrix.model_copy(deep=True),
            )
        return ConfusionMetrixMetricsModel()

    def add(self, metrics_model: ConfusionMetrixMetricsModel, num_samples: int) -> None:
        if num_samples > 0:
            if self.accuracy is not None and metrics_model.accuracy is not None:
                self.accuracy += metrics_model.accuracy / num_samples

            if self.precision is not None and metrics_model.precision is not None:
                self.precision += metrics_model.precision / num_samples

            if self.recall is not None and metrics_model.recall is not None:
                self.recall += metrics_model.recall / num_samples

            if self.f1 is not None and metrics_model.f1 is not None:
                self.f1 += metrics_model.f1 / num_samples

    def set_f1(self) -> None:
        if (
            self.precision is not None
            and self.precision >= 0.0
            and self.recall is not None
            and self.recall >= 0.0
        ):
            self.f1 = statistics.harmonic_mean([self.precision, self.recall])


class MicroConfusionMetrixMetricsModel(BaseModel):
    intent_set_metrics: ConfusionMetrixMetricsModel = ConfusionMetrixMetricsModel()
    intent_counter_metrics: ConfusionMetrixMetricsModel = ConfusionMetrixMetricsModel()
    intent_list_metrics: ConfusionMetrixMetricsModel = ConfusionMetrixMetricsModel()
    slot_set_metrics: ConfusionMetrixMetricsModel = ConfusionMetrixMetricsModel()


class ScorerOuputModel(BaseModel):
    confusion_metrix_matrics_micro: MicroConfusionMetrixMetricsModel
    num_examples: int
    percentage_times_full_score: float
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
    win_rate: Optional[float] = None
    num_sequences_processed_win_rate: Optional[int] = None
    error_messages_win_rate: Optional[List[str]] = None
    num_failed_function_execution_list: Optional[List[int]] = None


class MetricsAggregationModel(BaseModel):
    micro: Dict[str, ConfusionMetrixMetricsModel] = dict()
    macro: Dict[str, ConfusionMetrixMetricsModel] = dict()
    categories: List[str] = []
    raw_data: Dict[str, List[ConfusionMetrixMetricsModel]] = dict()
