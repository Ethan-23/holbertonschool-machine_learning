#!/usr/bin/env python3
"""0x09. Transfer Learning"""

import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    X is a numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data, where m is the number of data points
    Y is a numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X
    Returns: X_p, Y_p
    - X_p is a numpy.ndarray containing the preprocessed X
    - Y_p is a numpy.ndarray containing the preprocessed Y
    """
    x = K.applications.inception_resnet_v2.preprocess_input(X)
    y = K.utils.to_categorical(Y, 10)
    return x, y

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    inputs = K.Input(shape=(32,32,3))
    inputs_resize = K.layers.Lambda(lambda image: K.backend.resize_images(
        image,
        height_factor=(288 // 32),
        width_factor=(288 // 32),
        data_format="channels_last"
    ))(inputs)

    base_model = K.applications.InceptionResNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(288, 288, 3)
    )

    x = base_model(inputs_resize, training=False)
    x = K.layers.GlobalAveragePooling2D()(x)
    x = K.layers.Dense(500, activation='relu')(x)
    x = K.layers.Dropout(0.3)(x)
    output = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, output)

    base_model.trainable = False
    optimizer = K.optimizers.Adam()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["acc"])
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=300, epochs=4, verbose=1)

    model.save('cifar10.h5')
