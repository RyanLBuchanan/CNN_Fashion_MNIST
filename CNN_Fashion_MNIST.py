# CNN Fashion MNIST from Advanced Computer Vision - Lazy Programmer
# Input by Ryan L Buchanan 27AUG20

from __future__ import print_function, division
from builtins import range 
# may need to update: pip install -U future

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization, Input

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Helper
def y2indicator(Y):
    N = len(Y)
    K = len(set(Y))
    I = np.zeros((N, K))
    I[np.arange(N), Y] = 1
    return I


# Get the data
data = pd.read_csv('data/fashion-mnist_train.csv')
data = data.as_matrix()
    


