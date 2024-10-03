import numpy as np
from initialize import array_for


def add_random():
    k = np.random.randint(0, 255, (28, 28))  # this will play the role of our image
    # Keep it around, to add randomness to the dataset after
    input_init = array_for(k)  # We normalize it
    # it seems flatten will not work
    input_in = input_init.reshape((784, 1))
    print('random shape: ', input_in.shape)
    return input_in
