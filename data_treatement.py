import pandas as pd
import numpy as np

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('mnist_train.csv')  # this is our csv
data = np.array(df)
np.random.shuffle(data)


def split_into_batches(df, batch_size):
    """Split a DataFrame into batches of a specified size."""
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    # This is useful to make the calculus directly rather than using an if else
    # I keep it here, but I think it should be done during gradient descent rather than here
    return [df[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]


def data_treatment(data):
    m, n = data.shape
    # in this function, we assume we get a csv file, and want to split it in train and test
    # Shuffle the DataFrame
    # df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
    # Split the DataFrame into training and testing sets
    data_dev = data[0:1000].T  # that way they are ready to be fed to the NN
    y_dev = data_dev[0]  # labels
    x_dev = data_dev[1:n]
    x_dev = x_dev / 255.  # normalisation

    data_train = data[1000:m].T
    Y_train = data_train[0]  # we keep the labels
    X_train = data_train[1:n]
    X_train = X_train / 255.  # normalisation
    _, m_train = X_train.shape
    return x_dev, X_train, y_dev, Y_train


def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y


test_data, train_data, test_labels, train_labels = data_treatment(data)
test_labels_e = one_hot(test_labels)
train_labels_e = one_hot(train_labels)
print('train labels shape: ', train_labels_e.shape)
print('train labels: ', train_labels_e)
print('train data shape: ', train_data.shape)
print('train data: ', train_data)
