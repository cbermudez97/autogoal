from autogoal.kb import AlgorithmBase


class AlgorithmDistillerBase:
    def can_distill(self, algorithm: AlgorithmBase) -> bool:
        return False

    def distill(
        self, algorithm: AlgorithmBase, train_x, train_y, test_x, test_y,
    ) -> AlgorithmBase:
        raise NotImplementedError("The 'distill' method must be implemented")
