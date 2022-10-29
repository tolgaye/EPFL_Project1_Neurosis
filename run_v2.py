import numpy as np
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import math
from helpers import *
from implementations import *
import datetime
from src import *


def main():
    # Import data
    y_train, tx_train, ids_train, y_test, tx_test, ids_test =load("train.csv","test.csv")
    
    #Find Replacement for Out of Measurement Data and Replace
    x_clean_median = clean_data(tx_train, err_val=-999, find_replacement= np.median)
    x_clean = x_clean_median
    # Normalised version of the data (without the 1's column)
    x_normal = normalize(x_clean)
    
    first_col = np.ones((num_samples, 1))
    tx = np.concatenate((first_col, x_normal), axis=1)
    print(tx.shape)
    
    #Hyperparameters for Implementation
    lambda_ = 1e-5


    # Initialization
    
    print("Starting training with Ridge Regression...\n")
    
    RR_losses, RR_ws = ridge_regression(y_train,tx,lambda_)
    
    print(RR_losses)
    print("Training has succesfully ended..")
    
    
    print("\n\nGenerating .csv file...")

    # Open the test dataset
    y_sub, x_sub_raw, ids_sub = load_csv_data('data/test.csv')
     # Create the submission file
        
    create_csv_submission(ids_sub, y_sub, 'final-test.csv')

    print("\nfinal-test.csv file generated")


if __name__ == "__main__":
    main()
