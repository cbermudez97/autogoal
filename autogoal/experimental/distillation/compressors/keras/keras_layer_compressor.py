from typing import Callable
from numpy import log2
from autogoal.contrib.keras._generated import (
    Conv1D as AutoGOALConv1D,
    Conv2D as AutoGOALConv2D,
)
from tensorflow.keras.layers import (
    Layer,
    Conv1D,
    Conv2D,
    Conv3D,
    Bidirectional,
    Dense,
    ConvLSTM2D,
    Convolution1D,
    Convolution2D,
    Convolution3D,
    LocallyConnected1D,
    LocallyConnected2D,
    SeparableConv1D,
    SeparableConv2D,
    SeparableConvolution1D,
    SeparableConvolution2D,
    GRU,
    LSTM,
    SimpleRNN,
)
from tensorflow.keras.layers import deserialize
from tensorflow.python.keras.utils import generic_utils

from .utils import dispatcher


class _KerasLayerCompressor:
    def __init__(self, compression_ratio: float = 0.5):
        if compression_ratio <= 0 or compression_ratio > 1:
            raise ValueError("Param 'compression_ratio' must be in the interval (0,1]")
        self.ratio = compression_ratio

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

    @dispatcher.when(AutoGOALConv1D)
    def compress(self, layer: AutoGOALConv1D):
        config = layer.get_config()
        l2r = min(int(log2(self.ratio)), -1)
        config["filters"] += l2r
        return layer.__class__.from_config(config)

    @dispatcher.when(AutoGOALConv2D)
    def compress(self, layer: AutoGOALConv2D):
        config = layer.get_config()
        l2r = min(int(log2(self.ratio)), -1)
        config["filters"] += l2r
        return layer.__class__.from_config(config)

    @dispatcher.when(Layer)
    def compress(self, layer: Layer):
        return self.default(layer)

    @dispatcher.when(Conv1D)
    def compress(self, layer: Conv1D):
        return self.compress_conv(layer)

    @dispatcher.when(Conv2D)
    def compress(self, layer: Conv2D):
        return self.compress_conv(layer)

    @dispatcher.when(Conv3D)
    def compress(self, layer: Conv3D):
        return self.compress_conv(layer)

    @dispatcher.when(Convolution1D)
    def compress(self, layer: Convolution1D):
        return self.compress_conv(layer)

    @dispatcher.when(Convolution2D)
    def compress(self, layer: Convolution2D):
        return self.compress_conv(layer)

    @dispatcher.when(Convolution3D)
    def compress(self, layer: Convolution3D):
        return self.compress_conv(layer)

    @dispatcher.when(ConvLSTM2D)
    def compress(self, layer: ConvLSTM2D):
        return self.compress_conv(layer)

    @dispatcher.when(LocallyConnected1D)
    def compress(self, layer: LocallyConnected1D):
        return self.compress_conv(layer)

    @dispatcher.when(LocallyConnected2D)
    def compress(self, layer: LocallyConnected2D):
        return self.compress_conv(layer)

    @dispatcher.when(SeparableConv1D)
    def compress(self, layer: SeparableConv1D):
        return self.compress_conv(layer)

    @dispatcher.when(SeparableConv2D)
    def compress(self, layer: SeparableConv2D):
        return self.compress_conv(layer)

    @dispatcher.when(SeparableConvolution1D)
    def compress(self, layer: SeparableConvolution1D):
        return self.compress_conv(layer)

    @dispatcher.when(SeparableConvolution2D)
    def compress(self, layer: SeparableConvolution2D):
        return self.compress_conv(layer)

    def compress_conv(self, layer):
        config = layer.get_config()
        config["filters"] *= self.ratio
        return layer.__class__.from_config(config)

    @dispatcher.when(Bidirectional)
    def compress(self, layer: Bidirectional):
        config = layer.get_config()
        inner_config = deserialize(config["layer"])
        compressed_inner = self.compress(inner_config)
        config["layer"] = compressed_inner.get_config()
        backward_inner = config.pop("backward_layer", None)
        if backward_inner is not None:
            backward_layer = deserialize(backward_inner)
            compressed_backward_layer = self.compress(backward_layer)
            config["backward_layer"] = generic_utils.serialize_keras_object(
                compressed_backward_layer
            )
        return layer.__class__.from_config(config)

    @dispatcher.when(Dense)
    def compress(self, layer: Dense):
        return self.compress_units(layer)

    @dispatcher.when(GRU)
    def compress(self, layer: GRU):
        return self.compress_units(layer)

    @dispatcher.when(LSTM)
    def compress(self, layer: LSTM):
        return self.compress_units(layer)

    @dispatcher.when(SimpleRNN)
    def compress(self, layer: SimpleRNN):
        return self.compress_units(layer)

    def compress_units(self, layer):
        config = layer.get_config()
        config["units"] *= self.ratio
        return layer.__class__.from_config(config)

    def default(self, layer):
        return layer.__class__.from_config(layer.get_config())
