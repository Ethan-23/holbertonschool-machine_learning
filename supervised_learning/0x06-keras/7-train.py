#!/usr/bin/env python3
"""Sequential"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    network is the model to train
    data is a numpy.ndarray of shape (m, nx) containing the
        input data
    labels is a one-hot numpy.ndarray of shape (m, classes) containing the
        labels of data
    batch_size is the size of the batch used for mini-batch gradient descent
    epochs is the number of passes through data for mini-batch gradient descent
    verbose is a boolean that determines if output should be printed during
        training
    shuffle is a boolean that determines whether to shuffle the batches
        every epoch. Normally, it is a good idea to shuffle, but for
        reproducibility, we have chosen to set the default to False.
    validation_data is the data to validate the model with, if not None
    early_stopping is a boolean that indicates whether early stopping
        should be used
    patience is the patience used for early stopping
    learning_rate_decay is a boolean that indicates whether learning
        rate decay should be used
    alpha is the initial learning rate
    decay_rate is the decay rate
    Returns: the History object generated after training the model
    """
    def perEpoch(epoch):
        """
        calcs per epoch
        """
        return alpha / (1 + decay_rate * epoch)
    callbacks = []
    stop = K.callbacks.EarlyStopping(monitor='val_loss', mode='min',
                                     patience=patience)
    lr_Decay = K.callbacks.LearningRateScheduler(perEpoch, verbose=1)

    if validation_data and early_stopping:
        callbacks.append(stop)
    if validation_data and learning_rate_decay:
        callbacks.append(lr_Decay)
    return network.fit(data, labels, epochs=epochs,
                       batch_size=batch_size, verbose=verbose,
                       shuffle=shuffle, callbacks=callbacks,
                       validation_data=validation_data)
