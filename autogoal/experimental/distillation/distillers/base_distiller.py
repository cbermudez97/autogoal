import abc
from typing import List

from autogoal.kb import AlgorithmBase


class AlgorithmDistillerBase(abc.ABC):
    @abc.abstractmethod
    def can_distill(self, algorithm: AlgorithmBase) -> bool:
        pass

    @abc.abstractmethod
    def distill(
        self,
        algorithm: AlgorithmBase,
        train_inputs: dict,
        test_inputs: dict,
        registry: List,
    ) -> AlgorithmBase:
        pass
