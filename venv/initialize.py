import numpy as np
def createNNetwork(nb_layers, input_size, output_size):

    # The plan is to first create a neural netwok with only the input and output sizes changing.
    # I'll make it more complex later
    # i intend to first test 28*28 images (MINST database)
    hidden = 16  # for now there will be two hidden layers of 16 neurons.
    W1 = np.random.random((hidden, input_size))
    B1 = np.random.random((16, 1))

    W2 = np.random.random((hidden, hidden))
    B2 = np.random.random((16, 1))

    W3 = np.random.random((output_size, hidden))
    B3 = np.random.random((output_size, 1))

    # with this approach we are building the arrays for the weights and biases of each layer

    return W1, W2, W3, B1, B2, B3


