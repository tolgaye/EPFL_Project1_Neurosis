import numpy as np
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import math
from helpers import *
from implementations import *
import datetime
from src import *
from fold import *
from preprocess import *

def ridge_regression_sets(x, y, lambda_, k):
    # Create lists to save the different ws, accuracies and losses for the
    # different subsets we run the training on
    ws = []
    te_accs = []
    tr_accs = []
    te_losses = []
    tr_losses = []

    # Get the mask with the rows belonging to each subset
    x_jet_indexes = get_jet_indexes(x)

    # Iterate over the different subsets
    for i, indexes in enumerate(x_jet_indexes):
        # Get the rows relative to the i-th subset taken in consideration
        tx_i = prepare_x(x, x_jet_indexes, i)
        y_i = y[indexes]

        # Get indices for cross-validation
        k_indices = build_k_indices(y_i, k, 1)

        # Perform the training on the given-subset using cross validation and Ridge Regression
        w, tr_loss, te_loss, tr_acc, te_acc = \
            cross_validation(y_i, tx_i, k_indices, k,
                             lambda_, ridge_regression)

        # Add the results of the training of the i-th subset
        ws.append(w)
        te_accs.append(te_acc * tx_i.shape[0])
        tr_accs.append(tr_acc * tx_i.shape[0])
        te_losses.append(te_loss * tx_i.shape[0])
        tr_losses.append(tr_loss * tx_i.shape[0])

    # Compute the mean results from all the subsets
    mean_tr_loss = sum(tr_losses) / x.shape[0]
    mean_te_loss = sum(te_losses) / x.shape[0]
    mean_tr_acc = sum(tr_accs) / x.shape[0]
    mean_te_acc = sum(te_accs) / x.shape[0]

    return ws, mean_tr_loss, mean_te_loss, mean_tr_acc, mean_te_acc


def main():
    # Import data
    y_train, tx_train, ids_train, y_test, tx_test, ids_test =load("train.csv","test.csv")
    
    #Find Replacement for Out of Measurement Data and Replace
    x_clean_median = clean_data(tx_train, err_val=-999, find_replacement= np.median)
    x_clean = x_clean_median
    # Normalised version of the data (without the 1's column)
    x_normal = normalize(x_clean)
    
    num_samples = tx_train.shape[0]
    num_features = tx_train.shape[1]
    
    first_col = np.ones((num_samples, 1))
    tx = np.concatenate((first_col, x_normal), axis=1)
    print(tx.shape)
    
    #Hyperparameters for Implementation
    lambda_ = 1e-5
    k = 5


    # Initialization
    w_initial = np.ones((31,))
    
    print("Starting training with Ridge Regression...\n")
    
    LR_SGD_losses, LR_SGD_ws =  ridge_regression(y_train,tx,lambda_)
    
    print("Training has succesfully ended..")
    
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
