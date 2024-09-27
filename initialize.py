import numpy as np
import pandas as pd
import seaborn as sns
import math
from matplotlib import pyplot as plt
from pathlib import Path

import pandas as pd

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('data.csv')  # this is our csv

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
    # looking into dot function will be useful
    print('in function')
    l1 = np.matmul(w1, input) + b1
    print('l1: ', l1.shape)
    s1 = sigma_m(l1)
    l2 = np.matmul(w2, s1) + b2
    print('l2: ', l2.shape)
    s2 = sigma_m(l2)
    l3 = np.matmul(w3, s2) + b3
    # last activation function should be softmax
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

def data_treatement(csv):
    # in this function, we assume we get a csv file, and want to split it in train and test
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the DataFrame into training and testing sets
    train_ratio = 0.8
    train_size = int(len(df_shuffled) * train_ratio)

    train_data = df_shuffled[:train_size]
    test_data = df_shuffled[train_size:]
    return train_data, test_data


def split_into_batches(df, batch_size):
    """Split a DataFrame into batches of a specified size."""
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    # This is useful to make the calculus directly rather than using an if else
    return [df[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]

def trainning(train_data, batch_size, learning_rate, num_epoch, network):
    # this function should deal with the repetion process
    # to process each batch, be need to separate our data in batches
    # we are going to assume the data has been checked before and already shuffled
    batches = split_into_batches(train_data, batch_size)


    for i in range(num_epoch):
        # the whole process need to be repeated times the number of epochs
        # now, in here, we only need to figure how to process each batch
        for j in range(len(batches)):

            # in here the magic of backpropagation happens.

    return 5
