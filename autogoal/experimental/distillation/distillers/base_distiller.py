import abc
from typing import Any, Dict, List, Type
from autogoal.experimental.distillation.compressors.base_compressor import (
    ModelCompressorBase,
)

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
        registry: List[Type[ModelCompressorBase]],
        compressors_kwargs: Dict[Type[ModelCompressorBase], Dict[str, Any]] = {},
    ) -> AlgorithmBase:
        pass
