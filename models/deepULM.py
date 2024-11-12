"""deepULM"""

import numpy as np
import tensorflow as tf
from keras.layers import (
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    LeakyReLU,
    MaxPooling2D,
    UpSampling2D,
)
from keras.models import Model


def deepULM(
    input_shape,
    fdim=4,
    N_maxpool=4,
    k_enc=3,
    k_dec=3,
    dropout_rate=0.5,
    skip=False,
    skip_sum=False,
    depth=4,
    dilation=1,
    upscale=1,
    activation="relu",
):
    """deepULM model"""
    inputs = Input(shape=input_shape)

    def enc_layer(x, fdim, dilation=1, maxpool=True):
        x = Conv2D(fdim, (k_enc, k_enc), padding="same", dilation_rate=dilation)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2D(fdim, (k_enc, k_enc), padding="same", dilation_rate=dilation)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        if maxpool:
            x = MaxPooling2D((2, 2))(x)
        return x

    def dec_layer(x, fdim, upsample=True):
        if upsample:
            x = UpSampling2D(size=(2, 2))(x)

        x = Conv2DTranspose(fdim, (k_dec, k_dec), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        x = Conv2DTranspose(fdim, (k_dec, k_dec), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        return x

    # Upsampling
    if upscale != 1:
        output = UpSampling2D(
            size=(upscale, upscale), interpolation="nearest", name="Upsample"
        )(inputs)
    else:
        output = inputs

    max_resize_count = N_maxpool
    resize_points = np.round(np.linspace(0, depth - 1, max_resize_count))
    resize_points = resize_points.astype("uint8")

    # Pad the model if needed
    pad_x = output.shape[1] % 2**N_maxpool
    pad_y = output.shape[2] % 2**N_maxpool
    if pad_x or pad_y:
        oldshape = output.shape
        newshape_x = oldshape[1] + (2**N_maxpool - pad_x)
        newshape_y = oldshape[2] + (2**N_maxpool - pad_y)
        output = tf.image.resize_with_crop_or_pad(output, newshape_x, newshape_y)

    # Encoder
    enc_layers = []
    enc_layers.append(output)
    for i in range(depth):

        if i in resize_points:
            resize = True
            fdim = fdim * 2
        else:
            resize = False

        enc_layers.append(
            enc_layer(enc_layers[-1], fdim, dilation=dilation, maxpool=resize)
        )

    # Latent space
    latent = enc_layer(enc_layers[-1], fdim, maxpool=False)
    if dropout_rate != 0:
        latent = Dropout(dropout_rate)(latent)

    # Decoder
    dec_layers = []
    dec_layers.append(latent)

    for i in reversed(range(depth)):

        if i in resize_points:
            resize = True
            fdim = fdim / 2
        else:
            resize = False

        dec_lay = dec_layer(dec_layers[-1], fdim, upsample=resize)

        if skip:
            dec_lay = Concatenate(axis=-1)([dec_lay, enc_layers[i]])
        if skip_sum:
            dec_lay = Add()([dec_lay, enc_layers[i]])

        dec_layers.append(dec_lay)

    output = Conv2D(1, (3, 3), activation=activation, padding="same")(dec_layers[-1])

    # Unpad model
    if pad_x or pad_y:
        output = tf.image.resize_with_crop_or_pad(output, oldshape[1], oldshape[2])

    return Model(inputs=inputs, outputs=output, name="deepULM")
