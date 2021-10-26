import abc


class ModelCompressorBase(abc.ABC):
    def __init__(self, compression_ratio: float):
        super().__init__()
        assert (
            0 < compression_ratio and compression_ratio <= 1
        ), "Param 'compression_ratio' must be on the interval (0,1]."
        self.ratio = compression_ratio

    @abc.abstractmethod
    def can_compress(self, model) -> bool:
        pass

    @abc.abstractmethod
    def compress(self, model):
        pass
