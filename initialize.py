import numpy as np
import seaborn as sns


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


def array_for(x):
    return np.array([xi/255 for xi in x])


k = np.random.random_integers(0, 255, (28, 28))
input_init = array_for(k)
