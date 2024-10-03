import numpy as np
import pandas as pd
import seaborn as sns
import math

p = 784


def sigma_m(x):
    # return np.array([(1 / (1 + math.exp(xi))) for xi in x])
    # This can't work, because I have an array of arrays
    return 1 / (1 + np.exp(-x))


def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    # we changed the function to avoid overflow
    # we add the 1e-8 at the end to avoid underflow
    return expZ / (np.sum(expZ, axis=0, keepdims=True) + 1e-8)


def relu(x):
    return np.maximum(0, x)


def createNNetwork(input_size, output_size):
    # The plan is to first create a neural network with only the input and output sizes changing.
    # I'll make it more complex later
    # I intend to first test 28*28 images (MNIST database)
    hidden = 16  # for now there will be two hidden layers of 16 neurons.
    w1 = np.random.rand(hidden, input_size) - 0.5
    b1 = np.random.rand(hidden, 1) - 0.5

    w2 = np.random.rand(hidden, hidden) - 0.5
    b2 = np.random.rand(hidden, 1) - 0.5

    w3 = np.random.rand(output_size, hidden) - 0.5
    b3 = np.random.rand(output_size, 1) - 0.5

    # with this approach we are building the arrays for the weights and biases of each layer

    return w1, w2, w3, b1, b2, b3


def forwardpropagation(input, w1, w2, w3, b1, b2, b3):
    # looking into dot function will be useful
    # apparently matmul is fine here
    l1 = np.matmul(w1, input) + b1
    s1 = relu(l1)

    l2 = np.matmul(w2, s1) + b2
    s2 = relu(l2)

    l3 = np.matmul(w3, s2) + b3
    # last activation function should be softmax
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





