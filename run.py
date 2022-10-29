import numpy as np

from implementations import ridge_regression
from src.logistic import *
from src.helpers import *

def main():
    # Import data
    y, x_raw, ids = load_csv_data('data/train.csv')

    # Deal with mass missing values
    x = clean_mass_feature(x_raw)

    # Set hyperparameters
    lambda_ = 1e-5
    k = 5

    print("Starting training with Ridge Regression...\n")

    # Run Ridge Regression
    ws, tr_loss, te_loss, tr_acc, te_acc = ridge_regression(
        x, y, lambda_)

    print("Train accuracy={tr_acc:.3f}, test accuracy={te_acc:.3f}".format(
        tr_acc=tr_acc, te_acc=te_acc))
    print("Train MSE={tr_loss:.3f}, test MSE={te_loss:.3f}".format(
        tr_loss=tr_loss, te_loss=te_loss))

    print("\n\nGenerating .csv file...")

    # Open the test dataset
    y_sub, x_sub_raw, ids_sub = load_csv_data('data/test.csv')

    # Get the mask with the rows belonging to each subset
    x_sub_jet_indexes = get_jet_indexes(x_sub_raw)

    # Deal with mass missing values
    x_sub = clean_mass_feature(x_sub_raw)

    # Iterate over the parameters of the models trained on the different
    # subsets, and predict the labels of each subset
    for i, w in enumerate(ws):
        # Prepare the feature of the i-th subset
        tx_sub = prepare_x(x_sub, x_sub_jet_indexes, i)

        # Predict the labels
        y_sub[x_sub_jet_indexes[i]] = predict_labels(
            ws[i], tx_sub, mode='linear')

    # Create the submission file
    create_csv_submission(ids_sub, y_sub, 'final-test.csv')

    print("\nfinal-test.csv file generated")


if __name__ == "__main__":
    main()