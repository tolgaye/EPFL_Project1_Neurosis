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
    
    num_samples = tx_train.shape[0]
    num_features = tx_train.shape[1]
    
    first_col = np.ones((num_samples, 1))
    tx = np.concatenate((first_col, x_normal), axis=1)
    print(tx.shape)
    
    #Hyperparameters for Implementation
    max_iter = 50    
    gamma = 0.08


    # Initialization
    w_initial = np.ones((31,))
    
    print("Starting training with Mean Square SGD...\n")
    
    LR_SGD_losses, LR_SGD_ws =  mean_squared_error_sgd(y_train,tx,w_initial,max_iter,gamma)
    
    print("Training has succesfully ended..")
    
    
    
