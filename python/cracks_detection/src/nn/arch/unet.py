import numpy as np
from keras.engine.training import Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, Concatenate, Activation, MaxPool2D, \
    BatchNormalization
from keras.models import Input
from keras.optimizers import Adam

IMAGE_SIZE = (256, 256, 3)

def prepare_input(image, channels):
    if channels == 1:
        image = np.reshape(image, image.shape + (channels,))
    image = np.reshape(image, (1,) + image.shape)
    image = np.clip(image, 0, 255)
    image = np.divide(image, 255)
    if channels == 3:
        return image
    return np.uint8(image)


def prepare_output(image):
    image = image[:, :, 0]
    image = np.clip(image, 0, 1)
    image = np.multiply(image, 255)
    return image


def conv_block(x, num_filters):
    x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.1)(x)

    x = Conv2D(num_filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


def build_model(weights_input=None):
    size = 256
    num_filters = [32, 48, 64, 96]
    inputs = Input((size, size, 3))

    skip_x = []
    x = inputs
    ## Encoder
    for f in num_filters:
        x = conv_block(x, f)
        skip_x.append(x)
        x = MaxPool2D((2, 2))(x)

    ## Bridge
    x = conv_block(x, num_filters[-1])

    num_filters.reverse()
    skip_x.reverse()
    ## Decoder
    for i, f in enumerate(num_filters):
        x = UpSampling2D((2, 2))(x)
        xs = skip_x[i]
        x = Concatenate()([x, xs])
        x = conv_block(x, f)

    ## Output
    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(lr=1e-4), loss="binary_crossentropy", metrics=["accuracy"])

    model.summary()

    if weights_input:
        model.load_weights(weights_input)

    return model
