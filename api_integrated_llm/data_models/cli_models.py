from enum import Enum


class CliMode(str, Enum):
    DEFAULT = "default"
    EVALUATOR = "eval"
    SCORER = "score"

    def __str__(self):
        return str(self.value)
