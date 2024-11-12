"""Classifier"""

import numpy as np
from keras.layers import Activation, BatchNormalization, Conv2D, Input, LeakyReLU
from keras.models import Model


def classifier(input_shape, kmax=11, kmin=3, depth=8):
    """Classifier model"""
    inputs = Input(shape=input_shape)

    def _round_up_to_odd(f):
        return np.ceil(f) // 2 * 2 + 1

    def _round_down_to_odd(f):
        return np.floor(f) // 2 * 2 + 1

    kernels = _round_down_to_odd(np.linspace(kmax, kmin, depth)).astype("int32")

    x = inputs
    for kernel in kernels:
        x = Conv2D(4, (kernel, kernel), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)

    output = Conv2D(1, (3, 3), padding="same")(x)
    output = Activation("sigmoid")(output)

    model = Model(inputs=inputs, outputs=output, name="classification")
    return model
