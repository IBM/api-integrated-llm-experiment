from enum import Enum
from typing import Dict, List

from pydantic import BaseModel


class DatasetCategory(str, Enum):
    BIRD = "bird"
    COSQL = "cosql"
    SPARC = "sparc"

    def __str__(self):
        return str(self.value)


class ComputeCategory(str, Enum):
    AGENT = "agent"

    def __str__(self):
        return str(self.value)


class SubtaskCategory(str, Enum):
    SEQUENCING = "sequencing"
    SLOT_FILLING = "slot-filling"
    SINGLE_INTENT = "rest"

    def __str__(self):
        return str(self.value)


class DefaultMetricsAggregationConfiguration(BaseModel):
    dataset: List[str] = list(DatasetCategory)
    subtask: List[str] = list(SubtaskCategory)

    def get_dict(self) -> Dict[str, List[str]]:
        return {
            "dataset": [str(category) for category in self.dataset],
            "subtask": [str(category) for category in self.subtask],
        }
