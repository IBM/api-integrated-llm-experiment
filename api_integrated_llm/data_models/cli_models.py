from enum import Enum


class CliModeModel(str, Enum):
    DEFAULT = "default"
    EVALUATOR = "evaluator"
    SCORER = "scorer"
    PARSER = "parser"
    METRICS_AGGREGATOR = "metrics_aggregator"

    def __str__(self):
        return str(self.value)
