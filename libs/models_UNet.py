from keras import layers, models
import math


# ── encoder block ─────────────────────────────────────────────────────────────
# Returns (skip, downsampled)
# skip        → passed to the matching decoder block via concatenation
# downsampled → passed to the next encoder block

def encoder_block(x, filters, activation, downsample=True):
    shortcut = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = activation()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, shortcut])
    x = layers.BatchNormalization()(x)
    x = activation()(x)

    skip = x  # save before downsampling

    if downsample:
        x = layers.Conv2D(filters, (3, 3), strides=(2, 2), padding='same')(x)  # strided conv instead of MaxPooling2D

    return skip, x


def bottleneck(x, filters, activation):
    shortcut = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = activation()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, shortcut])
    x = layers.BatchNormalization()(x)
    x = activation()(x)

    return x


def decoder_block(x, skip, filters, activation):
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)  # replaces UpSampling2D

    x = layers.Concatenate()([x, skip])  # UNet skip connection

    shortcut = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = activation()(x)
    x = layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = layers.Add()([x, shortcut])
    x = layers.BatchNormalization()(x)
    x = activation()(x)

    return x


# ── build ─────────────────────────────────────────────────────────────────────

def build(
    size=128,
    basef=64,
    maxf=512,
    num_classes=6,
    image_channels=3,
    use_elevation=True,
    activation=layers.ReLU,
):

    # Inputs
    input_img = layers.Input((size, size, image_channels), name='image_input')
    if use_elevation:
        if image_channels != 3:
            raise ValueError('use_elevation=True requires image_channels=3')
        input_eleva = layers.Input((size, size, 1), name='elevation_input')
        x = layers.Concatenate(axis=-1, name='image_elevation_concat')([input_img, input_eleva])
        inputs = [input_img, input_eleva]
    else:
        x = input_img
        inputs = input_img

    # Filter progression  e.g. [64, 128, 256, 512]
    start_exp = int(math.log2(basef))
    end_exp   = int(math.log2(maxf))
    filters   = [2**i for i in range(start_exp, end_exp + 1)]

    depth = len(filters) - 1
    padded_size = 2**depth * math.ceil(size / 2**depth)
    pad = padded_size - size
    pad_top, pad_left = pad // 2, pad // 2
    pad_bottom, pad_right = pad - pad_top, pad - pad_left
    x = layers.ZeroPadding2D(((pad_top, pad_bottom), (pad_left, pad_right)))(x)

    # ── Encoder ──────────────────────────────────────────────────────────────
    skips = []
    for f in filters[:-1]:          # all but last → downsample
        skip, x = encoder_block(x, f, activation, downsample=True)
        skips.append(skip)

    # ── Bottleneck ───────────────────────────────────────────────────────────
    x = bottleneck(x, filters[-1], activation)

    # ── Decoder ──────────────────────────────────────────────────────────────
    for f, skip in zip(reversed(filters[:-1]), reversed(skips)):
        x = decoder_block(x, skip, f, activation)

    x = layers.Cropping2D(((pad_top, pad_bottom), (pad_left, pad_right)))(x)  # remove padding

    # ── Output head ──────────────────────────────────────────────────────────
    x = layers.Conv2D(num_classes, (1, 1), padding='same', activation='linear', name='seg_output')(x)

    return models.Model(inputs=inputs, outputs=x)
