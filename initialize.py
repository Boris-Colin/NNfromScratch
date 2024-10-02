import numpy as np
import pandas as pd
import seaborn as sns
import math
from matplotlib import pyplot as plt
from pathlib import Path

p = 784


def sigma_m(x):
    # return np.array([(1 / (1 + math.exp(xi))) for xi in x])
    # This can't work, because I have an array of arrays
    return 1 / (1 + np.exp(-x))


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def relu(x):
    return np.maximum(0, x)


def createNNetwork(input_size, output_size):

    # The plan is to first create a neural network with only the input and output sizes changing.
    # I'll make it more complex later
    # I intend to first test 28*28 images (MNIST database)
    hidden = 16  # for now there will be two hidden layers of 16 neurons.
    w1 = np.random.random((hidden, input_size))
    b1 = np.random.random((hidden, 1))

    w2 = np.random.random((hidden, hidden))
    b2 = np.random.random((hidden, 1))

    w3 = np.random.random((output_size, hidden))
    print('first: ', w3.shape)
    b3 = np.random.random((output_size, 1))

    # with this approach we are building the arrays for the weights and biases of each layer

    return w1, w2, w3, b1, b2, b3


def forwardpropagation(input, w1, w2, w3, b1, b2, b3):
    # looking into dot function will be useful
    # apparently matmul is fine here
    print('in function')
    print('input shape: ', input.shape)
    print('w1: ', w1.shape)
    print('b1: ', b1.shape)
    l1 = np.matmul(w1, input) + b1
    print('l1: ', l1.shape)
    s1 = relu(l1)

    print('w2: ', w2.shape)
    print('b2: ', b2.shape)
    l2 = np.matmul(w2, s1) + b2
    print('l2 shape: ', l2.shape)
    s2 = relu(l2)
    print('S2 shape: ', s2.shape)

    print('w3: ', w3.shape)
    print('b3: ', b3.shape)
    l3 = np.matmul(w3, s2) + b3
    # last activation function should be softmax
    print('l3: ', l3.shape)
    s3 = softmax(l3)
    # we need to return everything so that we can do back propagation correctly.
    return l1, s1, l2, s2, l3, s3


def array_for(x):
    return np.array([xi/255 for xi in x])


def add_random():
    k = np.random.randint(0, 255, (28, 28))  # this will play the role of our image
    # Keep it around, to add randomness to the dataset after
    input_init = array_for(k)  # We normalize it
    # it seems flatten will not work
    input_in = input_init.reshape((784, 1))
    print('random shape: ', input_in.shape)
    return input_in


"""w1, w2, w3, b1, b2, b3 = createNNetwork(p, 10)
print('w1: ', w1.shape)
print('w2: ', w2.shape)
print('w3: ', w3.shape)
print('b1: ', b1.shape)
print('b2: ', b2.shape)
print('b3: ', b3.shape)"""
# sx = forwardpropagation(input_in, w1, w2, w3, b1, b2, b3)
# print(sx)  # up until here it works how I wanted it to




