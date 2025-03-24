from __future__ import annotations
from typing import Dict, List, Set, Union
from pydantic import BaseModel


def get_file_name_without_extension(file_path: str) -> str:
    name = file_path.split("/")[-1]
    return ".".join(name.split(".")[:-1])


class SampleIgonoreModel(BaseModel):
    dataset: Dict[str, Set[Union[str, int]]] = dict()

    @staticmethod
    def get_sample_ignore_model(
        sample_dict: Dict[str, List[Union[str, int]]],
    ) -> SampleIgonoreModel:
        dataset: Dict[str, Set[Union[str, int]]] = dict()
        for file_path, ids in sample_dict.items():
            name = get_file_name_without_extension(file_path=file_path)
            dataset[name] = set(ids)
        return SampleIgonoreModel(dataset=dataset)

    def has_sample_to_ignore(self, file_path: str, sample_id: Union[str, int]) -> bool:
        name = get_file_name_without_extension(file_path=file_path)
        if name not in self.dataset:
            return False
        if sample_id not in self.dataset[name]:
            return False
        return True
