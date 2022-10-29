def get_jet_indexes(x: np.ndarray) -> List[np.ndarray]:
    """
    Gets the masks with the rows belonging to each of the jet number subsets.
    Arguments
    ---------
    x: np.ndarray
        The array to be indexed.
    Returns
    -------
    indexes: List[np.ndarray]
        A list of ndarrays (of booleans). Each ndarray contains the mask of the relative
        subset. `indexes[0]` contains has True values on the rows with jet number 0, `indexes[1]`
        has True values on the rows with jet number 1 and `indexes[2]` has True values on the
        rows with jet numbers 2 and 3.
    """

    return [
        x[:, 22] == 0,
        x[:, 22] == 1,
        np.bitwise_or(x[:, 22] == 2, x[:, 22] == 3)
    ]


"""
Contains the indexes of the rows to be removed for each subset, since they contain null values.
"""
jet_indexes = [
    [4, 5, 6, 12, 23, 24, 25, 26, 27, 28],  # indexes to be removed from rows with jet number 0
    [4, 5, 6, 12, 26, 27, 28],  # indexes to be removed from rows with jet number 1
    []  # no indexes are removed, since with jet numbers 2 and 3 no features are missing
]

def predict_labels(weights: np.ndarray, data: np.ndarray, mode: str = "logistic") -> np.ndarray:
    """
    Generates class predictions given weights, and a test data matrix.
    Arguments
    ---------
    weights: np.ndarray
        The weights of the predictive functions.
    data: np.ndarray
        The data for which the label must be predicted.
    mode: str
        The type of model, either 'logistic' (default) or 'linear'
    Returns
    -------
    y_pred: np.ndarray
        The predicted labels.
    """
    assert mode == "logistic" or "linear", "The model should be either logistic or linear"
    bound = 0 if mode == "linear" else 0.5
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= bound)] = -1 if mode == "linear" else 0
    y_pred[np.where(y_pred > bound)] = 1

    return y_pred

def clean_mass_feature(x: np.ndarray) -> np.ndarray:
    """
    Deals with the fact that some mass entries are null. It first creates a new column
    where each element is 0 if the corresponding row has a non-null (!=-999) value,
    otherwise it is 1. Then it substitutes the null values in the original mass column
    and adds the newly created column to the original dataset.
    Arguments
    ---------
    x: np.ndarray
        The dataset whose masses features need to be fixed.
    Returns
    -------
    x: np.ndarray
        The array with fixed mass values and with the new column.
    """
    # Create a new array containing all 0
    x_mass = np.zeros(x.shape[0])

    # Set the elements corresponding to rows with missing mass to 1
    x_mass[x[:, 0] == -999] = 1

    # Set the missing values to the median
    x[:, 0][x[:, 0] == -999] = np.median(x[:, 0][x[:, 0] != -999])

    # Add the newly created column to the dataset
    x = np.column_stack((x, x_mass))

    return x

def prepare_x(x: np.ndarray, indexes: List[np.ndarray], i: int):
    """
    Prepares the i-th subset of x to be used for training. In particular, it:
    * deletes the columns with missing values.
    * takes the logarithm of each feature.
    * standardizes the data.
    * builds the 2nd degree expansion of the rows.
    * adds a 1s column to be multiplied with a w_0 weight.
    Arguments
    ---------
    x: np.ndarray
        The dataset whose masses features need to be fixed.
    indexes: np.ndarray
        the mask containing the row indexes to be taken in consideration.
    i: int
        The subset to take in consideration.
    Returns
    -------
    x: np.ndarray
        The array ready to be used for training.
    """
    # Get the rows relative to the i-th subset taken in consideration
    tx_i = x[indexes[i]]

    # Delete the columns that are -999 for the given subset
    tx_del = np.delete(tx_i, jet_indexes[i], axis=1)

    # Take the logarithm of each column
    for li in range(tx_del.shape[1]):
        tx_del[:, li] = np.apply_along_axis(lambda n: np.log(
            1 + abs(tx_del[:, li].min()) + n), 0, tx_del[:, li])

    # Standardize the data
    tx_std = standardize(tx_del)[0]

    # Build the polynomial expansion of degree 2 and add the 1s column
    tx = build_poly_matrix_quadratic(tx_std)
    tx = np.c_[np.ones((tx.shape[0], 1)), tx]

    return tx

def create_csv_submission(ids: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    """
    Creates an output file in csv format for submission to Kaggle.
    Arguments
    ---------
    ids: np.ndarray
        Event ids associated with each prediction.
    y_pred: np.ndarray
        Predicted class labels.
    name: np.ndarray
        String name of .csv output file to be created.
    """

    y_pred[np.where(y_pred == 0)] = -1

    with open(name, 'w') as csv_file:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csv_file, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})
