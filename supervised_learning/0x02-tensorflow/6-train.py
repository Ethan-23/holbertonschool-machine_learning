#!/usr/bin/env python3
"""6. Train"""

import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    X_train is a numpy.ndarray containing the training input data
    Y_train is a numpy.ndarray containing the training labels
    X_valid is a numpy.ndarray containing the validation input data
    Y_valid is a numpy.ndarray containing the validation labels
    layer_sizes is a list containing the number of nodes in each
        layer of the network
    activations is a list containing the activation functions for
        each layer of the network
    alpha is the learning rate
    iterations is the number of iterations to train over
    save_path designates where to save the model
    Returns: the path where the model was saved
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection("y_pred", y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection("accuracy", accuracy)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection("loss", loss)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection("train_op", train_op)

    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    with tf.Session() as sees:
        sees.run(init)
        for i in range(iterations):
            if i % 100 == 0:
                train_cost = sees.run(loss, feed_dict={x: X_train, y: Y_train})
                train_accuracy = sees.run(accuracy,
                                          feed_dict={x: X_train, y: Y_train})
                valid_cost = sees.run(loss, feed_dict={x: X_valid, y: Y_valid})
                valid_accuarcy = sees.run(accuracy,
                                          feed_dict={x: X_valid, y: Y_valid})
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(train_cost))
                print("\tTraining Accuracy: {}".format(train_accuracy))
                print("\tValidation Cost: {}".format(valid_cost))
                print("\tValidation Accuracy: {}".format(valid_accuarcy))
            sees.run(train_op, feed_dict={x: X_train, y: Y_train})
        i += 1
        train_cost = sees.run(loss, feed_dict={x: X_train, y: Y_train})
        train_accuracy = sees.run(accuracy, feed_dict={x: X_train, y: Y_train})
        valid_cost = sees.run(loss, feed_dict={x: X_valid, y: Y_valid})
        valid_accuarcy = sees.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        print("After {} iterations:".format(i))
        print("\tTraining Cost: {}".format(train_cost))
        print("\tTraining Accuracy: {}".format(train_accuracy))
        print("\tValidation Cost: {}".format(valid_cost))
        print("\tValidation Accuracy: {}".format(valid_accuarcy))
        save = saver.save(sees, save_path)
        sees.close()
    return save
