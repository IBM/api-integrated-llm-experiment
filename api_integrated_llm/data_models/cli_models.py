from enum import Enum


class CliModeModel(str, Enum):
    DEFAULT = "default"
    EVALUATOR = "evaluator"
    SCORER = "scorer"
    PARSER = "parser"

    def __str__(self):
        return str(self.value)
