from enum import Enum


class CliModeModel(str, Enum):
    DEFAULT = "default"
    EVALUATOR = "evaluator"
    SCORER = "scorer"

    def __str__(self):
        return str(self.value)
