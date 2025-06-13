import numpy as np

#Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Derivative of sigmoid activation function
def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

#Mean squared error for loss calcuation
def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()