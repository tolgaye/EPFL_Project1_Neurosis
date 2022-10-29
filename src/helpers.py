import csv
import numpy as np
from typing import Tuple, List

def load_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]
    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]
    return yb, input_data, ids

def load(trainFile, testFile):
    """
    Builds various numpy arrays from the given .csv format training 
    and test tests.
    Args:
        trainFile: file name/path for the input training set
        testFile: file name/path for the input test set
    Returns: 
        y_train: labels in the training set as a numpy array
        tx_train: features in the training set as a numpy array
        ids_train: ids of the training data points as a numpy array
        y_test: labels in the test set as a numpy array
        tx_test: features in the test set as a numpy array
        ids_test: ids of the test data points as a numpy array
    """
    print('\nLoading the raw training and test set data...')
    y_train, tx_train, ids_train = load_data(trainFile)
    y_test, tx_test, ids_test = load_data(testFile)
    print('\n... finished.')
    return y_train, tx_train, ids_train, y_test, tx_test, ids_test

def most_frequent(x, extended_output=False):
    """
    Get the most frequent value in an array
    """
    counter = {}
    max_val = x[0]
    counter[max_val] = 1
    for val in x[1:]:
        if(val in counter):
            counter[val] += 1
            if counter[val] > counter[max_val]:
                max_val = val
        else:
            counter[val] = 1
                
    return (max_val, counter[max_val]) if extended_output else max_val

def replace(x, err_val, find_replacement):
    """
    Replace each occurence of a specified value in an array
    according to a specified replacement function
    """
    replacement = find_replacement(x[x != err_val])

    replaced = x.copy()
    replaced[replaced == err_val] = replacement
    
    return replaced
   
def clean_data(x, err_val, find_replacement):
    """
    Clean a matrix by replacing errors values in each column
    according to a specified replacement function
    """
    x_clean = np.zeros(x.shape)
    nb_features = x.shape[1]
    
    for feature in range(nb_features):
        x_clean[:, feature] = replace(x[:, feature], err_val, find_replacement)
        
    return x_clean

def preprocess(x, to_replace, do_normalise=True, augment_param=None):
    """
    Preprocess the data matrix
    1. to_replace clean a matrix by replacing errors values in each column
    according to a specified replacement function.
    e.g. [(-111, 'mean')] will replace all occurances of -111 with the mean value
    of that featrue value over all samples (excluding -111).
    2. do_normalise normalises the data if set true
    3. add_bias adds a column of ones for the bias term
    """

    replace_method_map = {
        'mean': np.mean,
        'most_frequent': most_frequent,
        'median': np.median
    }

    # for each err_val to be replaced (to_replace), replace
    # with corresponding replace method
    for err_val, replace_method in to_replace:
        find_replacement = replace_method_map[replace_method]
        x = clean_data(x, err_val, find_replacement)

    if do_normalise:
        x = normalise(x)

    if augment_param:
        x = fea.augment_features(x, augment_param)
    
    return 


def normalize(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardizes the original data set.
    Parameters
    ----------
    x: ndarray
        Matrix that contains the data points to be standardized.
    Returns
    -------
    x: np.ndarray
        The standardized dataset
    mean_x: np.ndarray
        The mean of x before the standardization
    mean_x: np.ndarray
        The standard deviation of x before the standardization
    """
    mean_x = np.mean(x)
    x = x - mean_x
    std_x = np.std(x)
    x = x / std_x
    return x

from typing import Tuple
import numpy as np


def split_data(x: np.ndarray, y: np.ndarray, ratio: float, seed: float = 1) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    Parameters
    ----------
    x : np.ndarray
        The array containing the features.
    y : np.ndarray
        The array containing the targets
    ratio: float
        The ratio between the number of wanted training data points and
        the total number of data points.
    seed: float
        The seed with which the random generator is initialized.
    Returns
    -------
    x_train, y_train, x_test, y_test : ndarray
    """
    # Set seed
    np.random.seed(seed)

    # Randomly choose indexes of train set
    data_len = x.shape[0]
    idxs = np.random.choice(data_len, size=round(
        data_len * ratio), replace=False)

    # Create a mask from indexes
    mask = np.zeros(data_len, dtype=bool)
    mask[idxs] = True

    #      x_train  y_train  x_test    y_test
    return x[mask], y[mask], x[~mask], y[~mask]
