import numpy as np
from data_treatement import m, one_hot


def relu_deriv(Z):
    # before it returned a boolean, which was fine in most cases, but this is more accurate
    return np.where(Z > 0, 1, 0)


def backward_prop(Z1, A1, Z2, A2, Z3, A3, w1, w2, w3, train_data, train_labels):
    one_hot_data = one_hot(train_labels)
    dZ3 = A3 - one_hot_data  # (output_size, m)
    dW3 = (1 / m) * np.dot(dZ3, A2.T)  # (output_size, hidden)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)  # (output_size, 1)

    dA2 = np.dot(w3.T, dZ3)  # (hidden, m)
    dZ2 = dA2 * relu_deriv(Z2)  # (hidden, m)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)  # (hidden, hidden)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # (hidden, 1)

    dA1 = np.dot(w2.T, dZ2)  # (hidden, m)
    dZ1 = dA1 * relu_deriv(Z1)  # (hidden, m)
    dW1 = (1 / m) * np.dot(dZ1, train_data.T)  # (hidden, input_size)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W3 - alpha * dW3
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3
