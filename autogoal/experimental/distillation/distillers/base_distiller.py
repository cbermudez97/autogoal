import abc
from typing import List

from autogoal.kb import AlgorithmBase


class AlgorithmDistillerBase(abc.ABC):
    def __init__(self, compression_ratio: float) -> None:
        super().__init__()
        assert (
            0 < compression_ratio and compression_ratio <= 1
        ), "Param 'compression_ratio' must be on the interval (0,1]."
        self.ratio = compression_ratio

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
