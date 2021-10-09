from autogoal.experimental.distillation.compressors.base_compressor import (
    ModelCompressorBase,
)
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model

from .keras_layer_compressor import _KerasLayerCompressor


class KerasModelCompressor(ModelCompressorBase):
    def __init__(self, ratio=0.5):
        self.ratio = ratio
        self.layer_compressor = _KerasLayerCompressor(ratio=ratio)

    def can_compress(self, model):
        return isinstance(model, Model)

    def compress(self, model: Model):
        compressed_model = clone_model(
            model,
            clone_function=self.layer_compressor(
                is_output=lambda x: (x.name in model.output_names),
            ),
        )
        return compressed_model
