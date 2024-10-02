import numpy as np
from data_treatement import m


def relu_deriv(Z):
    return Z > 0


def backward_prop(Z1, A1, Z2, A2, Z3, A3, w1, w2, w3, train_data, train_labels_e):
    dZ3 = A3 - train_labels_e
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    dZ2 = w3.T.dot(dZ3) * relu_deriv(Z2)
    # this is for the two first layers. I don't need to change it too much.
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = w2.T.dot(dZ2) * relu_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(train_data.T)  # need to change X
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    W3 = W2 - alpha * dW2
    b3 = b3 - alpha * db3
    return W1, b1, W2, b2, W3, b3