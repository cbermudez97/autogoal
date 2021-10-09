import abc

from autogoal.kb import AlgorithmBase


class AlgorithmDistillerBase(abc.ABC):
    @abc.abstractmethod
    def can_distill(self, algorithm: AlgorithmBase) -> bool:
        pass

    @abc.abstractmethod
    def distill(
        self, algorithm: AlgorithmBase, train_x, train_y, test_x, test_y,
    ) -> AlgorithmBase:
        pass
