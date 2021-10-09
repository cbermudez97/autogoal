class ModelCompressorBase:
    def can_compress(self, model) -> bool:
        return False

    def compress(self, model):
        raise NotImplementedError("The 'compress' method must be implemented")
