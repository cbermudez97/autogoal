from typing import Callable
from tensorflow.keras.layers import Layer, Dense, Conv1D, Conv2D, Conv3D

from .utils import dispatcher


class _KerasLayerCompressor:
    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, is_output: Callable = lambda x: False):
        def compressor_fn(layer: Layer) -> Layer:
            if not isinstance(layer, Layer):
                raise ValueError(
                    "'layer' argument must be an instance of tensorflow.keras.layers.Layer"
                )
            if is_output(layer):
                return self.default(layer)
            return self.compress(layer)

        return compressor_fn

    @dispatcher.on("layer")
    def compress(self, layer: Layer):
        pass

    @dispatcher.when(Layer)
    def compress(self, layer: Layer):
        return self.default(layer)

    @dispatcher.when(Dense)
    def compress(self, layer: Dense):
        config = layer.get_config()
        config["units"] *= self.ratio
        return layer.__class__.from_config(config)

    @dispatcher.when(Conv1D)
    def compress(self, layer: Conv1D):
        return self.compress_conv(layer)

    @dispatcher.when(Conv2D)
    def compress(self, layer: Conv2D):
        return self.compress_conv(layer)

    @dispatcher.when(Conv3D)
    def compress(self, layer: Conv3D):
        return self.compress_conv(layer)

    def compress_conv(self, layer):
        config = layer.get_config()
        config["filters"] *= self.ratio
        return layer.__class__.from_config(config)

    def default(self, layer):
        return layer.__class__.from_config(layer.get_config())
