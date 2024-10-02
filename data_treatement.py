import pandas as pd
import numpy as np

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('mnist_train.csv')  # this is our csv


def data_treatment(csv):
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


train_data, test_data = data_treatment(df)
batches = split_into_batches(train_data, 256)
print('batches shape: ', batches)
