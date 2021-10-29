import abc


class ModelCompressorBase(abc.ABC):
    @abc.abstractmethod
    def can_compress(self, model) -> bool:
        pass

    @abc.abstractmethod
    def compress(self, model):
        pass
