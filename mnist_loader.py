"""
Mnist_loader
A library to load the MNIST image data. For details of the
data structures that are returned, see the doc strings for ''load_data''
and ''load_data_wrapper''. In practice, ''load_data_wrapper''
is the function usually called by our neural network code.
"""
import numpy as np
import gzip
import pickle

def load_data():
    f = gzip.open('data\mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    f.close()
    return(training_data, validation_data, test_data)

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_results = [vectorized_result(y) for y in va_d[1]]
    validation_data = zip(validation_inputs, validation_results)
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    test_data = zip(test_inputs, test_results)
    return training_data, validation_data, test_data
    """Retrun a tuple containing"""

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere. This is used to convert a digit
    (0,...9) into a corresponding desired output from the neural network."""
    e = np.zeros((10,1))
    e[j] = 1.0
    return e
