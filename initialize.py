import numpy as np
import seaborn as sns
import math


def createNNetwork(input_size, output_size):

    # The plan is to first create a neural network with only the input and output sizes changing.
    # I'll make it more complex later
    # I intend to first test 28*28 images (MNIST database)
    hidden = 16  # for now there will be two hidden layers of 16 neurons.
    w1 = np.random.random((hidden, input_size))
    b1 = np.random.random((16, 1))

    w2 = np.random.random((hidden, hidden))
    b2 = np.random.random((16, 1))

    w3 = np.random.random((output_size, hidden))
    b3 = np.random.random((output_size, 1))

    # with this approach we are building the arrays for the weights and biases of each layer

    return w1, w2, w3, b1, b2, b3


def sigma_m(x):
    # return np.array([(1 / (1 + math.exp(xi))) for xi in x])
    # This can't work, because I have an array of arrays
    return 1 / (1 + np.exp(-x))


def forwardpropagation(input, w1, w2, w3, b1, b2, b3):
    print('in function')
    l1 = np.matmul(w1, input) + b1
    print('l1: ', l1.shape)
    s1 = sigma_m(l1)
    l2 = np.matmul(w2, s1) + b2
    print('l2: ', l2.shape)
    s2 = sigma_m(l2)
    l3 = np.matmul(w3, s2) + b3
    print('l3: ', l3.shape)
    s3 = sigma_m(l3)
    return s3


def array_for(x):
    return np.array([xi/255 for xi in x])


k = np.random.randint(0, 255, (28, 28))  # this will play the role of our image
input_init = array_for(k)  # We normalize it
print(input_init)
# it seems flatten will not work
input_in = input_init.reshape((784, 1))
print(input_in.shape)
p = 784

w1, w2, w3, b1, b2, b3 = createNNetwork(p, 10)
print('w1: ', w1.shape)
print('w2: ', w2.shape)
print('w3: ', w3.shape)
print('b1: ', b1.shape)
print('b2: ', b2.shape)
print('b3: ', b3.shape)
sx = forwardpropagation(input_in, w1, w2, w3, b1, b2, b3)
print(sx)
