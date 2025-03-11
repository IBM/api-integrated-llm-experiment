from __future__ import annotations
from enum import Enum
import statistics
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

from api_integrated_llm.data_models.source_models import (
    EvaluationOutputResponseDataUnit,
)


class WinRateResultUnitModel(BaseModel):
    valid: bool = False
    pred_function_calls: List[Union[str, Dict[str, Union[str, int, float]]]] = []
    gold_function_calls: List[Union[str, Dict[str, Union[str, int, float]]]] = []
    num_failed_function_execution: int = 0
    error_messages: List[str] = []

    def get_length_gold_function_calls(self) -> int:
        return len(self.gold_function_calls)

    def get_basic_rate_unit_model(self) -> BasicRateUnitModel:
        return (
            BasicRateUnitModel(success=1, fail=0)
            if self.valid
            else BasicRateUnitModel(success=0, fail=1)
        )


class WinRateResultModel(BaseModel):
    win_rate_result: List[WinRateResultUnitModel] = []

    def get_rate(self) -> Optional[float]:
        if len(self.win_rate_result) == 0:
            return None

        return (
            len(list(filter(lambda result: result.valid, self.win_rate_result)))
        ) / len(self.win_rate_result)


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


class BasicRateUnitModel(BaseModel):
    success: int = 0
    fail: int = 0

    def is_valid(self) -> bool:
        return (self.success + self.fail) > 0

    def add(self, unit_model: BasicRateUnitModel) -> None:
        self.success += unit_model.success
        self.fail += unit_model.fail

    def get_rate(self) -> Optional[float]:
        return (self.success) / (self.success + self.fail) if self.is_valid() else None

    def get_num_samples(self) -> int:
        return self.success + self.fail


class BasicRateModel(BaseModel):
    rate: Optional[float] = None
    unit_model: Optional[BasicRateUnitModel] = None

    @staticmethod
    def get_basic_rate_model(unit_model: BasicRateUnitModel) -> BasicRateModel:
        if unit_model.is_valid():
            rate_model = BasicRateModel()
            rate_model.rate = unit_model.get_rate()
            rate_model.unit_model = unit_model.model_copy(deep=True)
            return rate_model
        return BasicRateModel()

    def add_micro(self, rate_model: BasicRateModel) -> None:
        if rate_model.unit_model is not None and rate_model.unit_model.is_valid():
            if self.unit_model is None:
                self.unit_model = BasicRateUnitModel()
            self.unit_model.add(unit_model=rate_model.unit_model)
            self.rate = self.unit_model.get_rate()

    def add_macro(self, rate_model: BasicRateModel, num_samples: int) -> None:
        if num_samples <= 0:
            return

        if rate_model.rate is not None:
            if self.rate is None:
                self.rate = 0.0
            self.rate += rate_model.rate / num_samples

    def get_num_samples(self) -> Optional[int]:
        if self.unit_model is not None:
            return self.unit_model.get_num_samples()
        return None


class BasicRateDictModel(BaseModel):
    rate_dictionary: Dict[str, BasicRateModel] = dict()
    raw_data: Dict[str, List[WinRateResultUnitModel]] = dict()

    def add_micro(self, rate_dict_model: BasicRateDictModel) -> None:
        for key, rate_model in rate_dict_model.rate_dictionary.items():
            if rate_model.unit_model is not None:
                if key in self.rate_dictionary:
                    self.rate_dictionary[key].add_micro(rate_model=rate_model)
                else:
                    self.rate_dictionary[key] = rate_model.model_copy(deep=True)

    def add_macro(self, rate_dict_model: BasicRateDictModel, num_samples: int) -> None:
        if num_samples <= 0:
            return

        for key, rate_model in rate_dict_model.rate_dictionary.items():
            if rate_model.rate is not None:
                if key in self.rate_dictionary:
                    self.rate_dictionary[key].add_macro(
                        rate_model=rate_model, num_samples=num_samples
                    )
                else:
                    self.rate_dictionary[key] = BasicRateModel(
                        rate=0.0, unit_model=None
                    )
                    self.rate_dictionary[key].add_macro(
                        rate_model=rate_model, num_samples=num_samples
                    )


class BasicRateDictMetaModel(BaseModel):
    micro_rate: BasicRateDictModel = BasicRateDictModel()
    macro_rate: BasicRateDictModel = BasicRateDictModel()


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

    @staticmethod
    def get_confusion_matrix_metrics_micro_by_output_length(
        confusion_matrix_dict: Dict[int, ConfusionMatrixModel],
    ) -> Dict[int, ConfusionMetrixMetricsModel]:
        return {
            frequency: ConfusionMetrixMetricsModel.get_confusion_matrix_metrics_micro(
                confusion_matrix=metrics_model,
            )
            for frequency, metrics_model in confusion_matrix_dict.items()
        }

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


class MicroConfusionMetrixMetricsProblemLevelModel(BaseModel):
    intent_set_metrics_list: List[ConfusionMetrixMetricsModel] = []
    intent_counter_metrics_list: List[ConfusionMetrixMetricsModel] = []
    intent_list_metrics_list: List[ConfusionMetrixMetricsModel] = []
    slot_set_metrics_list: List[ConfusionMetrixMetricsModel] = []


class MicroConfusionMetrixMetricsModel(BaseModel):
    intent_set_metrics: ConfusionMetrixMetricsModel = ConfusionMetrixMetricsModel()
    intent_counter_metrics: ConfusionMetrixMetricsModel = ConfusionMetrixMetricsModel()
    intent_list_metrics: ConfusionMetrixMetricsModel = ConfusionMetrixMetricsModel()
    slot_set_metrics: ConfusionMetrixMetricsModel = ConfusionMetrixMetricsModel()


class MicroConfusionMetrixMetricsByOutputLengthModel(BaseModel):
    intent_set_metrics: Dict[int, ConfusionMetrixMetricsModel] = dict()
    intent_counter_metrics: Dict[int, ConfusionMetrixMetricsModel] = dict()
    intent_list_metrics: Dict[int, ConfusionMetrixMetricsModel] = dict()
    slot_set_metrics: Dict[int, ConfusionMetrixMetricsModel] = dict()


class MicroConfusionMetrixMetricsByOutputLengthProblemLevelModel(BaseModel):
    intent_set_metrics_list: Dict[int, List[ConfusionMetrixMetricsModel]] = dict()
    intent_counter_metrics_list: Dict[int, List[ConfusionMetrixMetricsModel]] = dict()
    intent_list_metrics_list: Dict[int, List[ConfusionMetrixMetricsModel]] = dict()
    slot_set_metrics_list: Dict[int, List[ConfusionMetrixMetricsModel]] = dict()


class ContentPairModel(BaseModel):
    gold: List[str] = []
    predicted: List[str] = []


class ScorerOuputModel(BaseModel):
    confusion_metrix_matrics_micro_problem_level: Optional[
        MicroConfusionMetrixMetricsProblemLevelModel
    ] = None
    confusion_metrix_matrics_micro_model_by_output_length_problem_level: Optional[
        MicroConfusionMetrixMetricsByOutputLengthProblemLevelModel
    ] = None
    confusion_metrix_matrics_micro: MicroConfusionMetrixMetricsModel
    confusion_metrix_matrics_micro_model_by_output_length: Optional[
        MicroConfusionMetrixMetricsByOutputLengthModel
    ] = None
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
    intent_pair_models: Optional[List[ContentPairModel]] = None
    slot_pair_models: Optional[List[ContentPairModel]] = None
    win_rate: Optional[float] = None
    num_sequences_processed_win_rate: Optional[int] = None
    error_messages_win_rate: Optional[List[str]] = None
    num_failed_function_execution_list: Optional[List[int]] = None
    win_rate_result_model: Optional[WinRateResultModel] = None


class MetricsAggregationModel(BaseModel):
    micro: Dict[str, ConfusionMetrixMetricsModel] = dict()
    macro: Dict[str, ConfusionMetrixMetricsModel] = dict()
    categories: List[str] = []
    raw_data: Dict[str, List[ConfusionMetrixMetricsModel]] = dict()


class MicroConfusionMetrixMetricsByOutputLengthContainerModel(BaseModel):
    intent_set_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = dict()
    intent_counter_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = dict()
    intent_list_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = dict()
    slot_set_metrics: Dict[str, List[ConfusionMetrixMetricsModel]] = dict()


class MetaMetricsAggregationModel(BaseModel):
    intent_set_metrics: MetricsAggregationModel = MetricsAggregationModel()
    intent_counter_metrics: MetricsAggregationModel = MetricsAggregationModel()
    intent_list_metrics: MetricsAggregationModel = MetricsAggregationModel()
    slot_set_metrics: MetricsAggregationModel = MetricsAggregationModel()
    win_rate_metrics: BasicRateDictMetaModel = BasicRateDictMetaModel()


class AggegatorOutputModel(BaseModel):
    aggregated_metrics: Dict[str, MetaMetricsAggregationModel] = dict()
