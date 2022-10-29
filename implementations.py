import numpy as np
import math
from typing import  Tuple
from random import randrange

from src.logistic import *
from src.gradient import *
from src.loss import *


def mean_squared_error_gd(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray,
                     max_iters: int, gamma: float) -> Tuple[float, np.ndarray]:
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y,tx,w)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws


def mean_squared_error_sgd(y: np.ndarray, tx: np.ndarray,initial_w: np.ndarray,
                     max_iters: int, gamma: float) -> Tuple[float, np.ndarray]:
    """Stochastic gradient descent."""
    # Define parameters to store w and loss
   
    ws = [initial_w]
    losses=[]
    w = initial_w
    
    for n_iter in range(max_iters):
        # compute loss, gradient
        rand_idx = np.random.randint(len(y))
        rand_tx = tx[rand_idx]
        rand_y = y[rand_idx]
        
        grad = compute_gradient(rand_y, rand_tx, w)  
        loss = compute_loss(y,tx,w)
        # gradient w by descent update
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws


def least_squares(y, tx):
    """
    Least squares regression using normal equations
    Args:
        y: labels
        tx: features
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    coefficient_matrix = tx.T.dot(tx)
    constant_vector = tx.T.dot(y)
    w = np.linalg.solve(coefficient_matrix, constant_vector)
    loss = compute_loss(y, tx, w)
    
    return loss, w


def ridge_regression(y, tx, lambda_):
    """
    Ridge regression using normal equations
    Args:
        y: labels
        tx: features
        lambda_: regularization parameter
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on mean squared error
    """
    coefficient_matrix = tx.T.dot(tx) + 2 * len(y) * lambda_ * np.identity(tx.shape[1])
    constant_vector = tx.T.dot(y)
    w = np.linalg.solve(coefficient_matrix, constant_vector)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using stochastic gradient descent 
    Args:
        y: labels
        tx: features
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on logistic loss
    """
 
    w = initial_w
    loss = None
    losses = []
    threshold = 1e-8
    for iter in range(max_iters):
        loss = compute_logistic_loss(y, tx, w)
        gradient_vector = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient_vector
        if iter % 1 == 0:
            print("Current iteration of GD={i}, loss={l}".format(i=iter, l=loss))
            if iter % 2000 == 0:
                # Adaptive learning rate
                gamma = gamma*0.1
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
        
    return w, losses


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent 
    Args:
        y: labels
        tx: features
        lambda_: regularization parameter
        initial_w: initial weight vector
        max_iters: number of steps to run
        gamma: step-size
    Returns:
        w: optimized weight vector for the model
        loss: optimized final loss based on logistic loss
    """
    threshold = 1e-8
    w = initial_w
    ws = [initial_w]
    losses = []
    w = initial_w
    for _ in range(max_iters):
        loss, gradient_vector = penalized_logistic_regression(y, tx, w, lambda_)
        w = w - gamma * gradient_vector
        ws.append(w)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break # convergence criterion met
    return ws[-1], losses[-1]
