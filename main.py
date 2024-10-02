import numpy as np
import pandas as pd
from data_treatement import one_hot, data_treatment
from initialize import forwardpropagation, createNNetwork
from training import backward_prop, update_params


p = 784

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('mnist_train.csv')  # this is our csv
data = np.array(df)
m, n = data.shape
np.random.shuffle(data)

test_data, train_data, test_labels, train_labels = data_treatment(data)
test_labels_e = one_hot(test_labels)
train_labels_e = one_hot(train_labels)

weights1, weights2, weights3, bias1, bias2, bias3 = createNNetwork(p, 10)
print(bias3.shape)

def reverse_hot(L):
    # A function to return the actual value predicted.
    maxi = np.max(L)
    # this doesn't work, I'll look into it later
    return np.where(L == maxi)[0]


def get_predictions(A2):
    """x = reverse_hot(A2)
    print('predicted values: ', x)"""
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def training(training_data, training_labels, batch_size, learning_rate, num_epoch, w1, w2, w3, B1, B2, B3):
    W1, b1, W2, b2, W3, b3 = w1, B1, w2, B2, w3, B3
    print('b3: ', b3.shape)
    # this function should deal with the repetion process
    # to process each batch, be need to separate our data in batches
    # we are going to assume the data has been checked before and already shuffled
    for i in range(num_epoch):
        # the whole process need to be repeated times the number of epochs
        # now, in here, we only need to figure how to process each batch
        # but I'll try that later
        z1, a1, z2, a2, z3, a3 = forwardpropagation(training_data, W1, W2, W3, b1, b2, b3)
        print('Forward propagation')
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(z1, a1, z2, a2, z3, a3, W1, W2, W3, training_data, training_labels)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate)
        print('epoch number ' + str(i) + ' completed')
        if i % 8 == 0:
            print('Iteration: ', i/8)
            predictions = get_predictions(a3)
            print(get_accuracy(predictions, training_labels))

    return W1, b1, W2, b2, W3, b3


W1, b1, W2, b2, W3, b3 = training(train_data, train_labels_e,2, 0.01, 100,
                                  weights1, weights2, weights3, bias1, bias2, bias3)
