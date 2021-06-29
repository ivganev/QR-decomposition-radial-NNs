import numpy as np
from typing import List
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from source import *


#################################
## Synthetic dataset based on the normal distribution.
## Input and output dimensions are both 2.
#################################

x_med = torch.zeros((121**2, 2))
y_med = torch.zeros((121**2, 2))

for i in range(121):
    for j in range(121):
        x_med[121*i + j] = torch.tensor([-3.0 + i/20, -3.0 + j/20])
        y_med[121*i + j] = torch.sigmoid(x_med[121*i + j])



#################################
## Instantiate a radial neural network
#################################

radnet = RadNet(eta=torch.sigmoid, dims=[2, 16, 64, 128, 16, 2], has_bias=False)


#################################
## Compute the reduced network (R)
## Check that loss functions match
#################################

radnet_red = radnet.reduced_network()

assert all([all(radnet(x_med)[0] - radnet_red(x_med)[0] < 0.0001), \
all(radnet(x_med)[1] - radnet_red(x_med)[1] < 0.0001), \
(loss_fn(radnet(x_med), y_med) - loss_fn(radnet_red(x_med), y_med) < 0.0001).item()]), \
"Loss functions do not match"
        

#################################
## Run the original model until the loss is below 0.01
## This usually takes more than 20 seconds
#################################

print(" ")
print("Training the original model:")

tic = time.time()

model_trained = training_loop_with_stop(
    n_epochs = 5000, 
    learning_rate = 0.01,
    model = radnet,
    params = list(radnet.parameters()),
    x_train = x_med,
    y_train = y_med,
    stopping_value=0.01,
    verbose=True)

toc = time.time()
print()
print("Elapsed time:", toc - tic)


#################################
## Run the reduced model until the loss is below 0.01
## This usually takes less than 15 seconds,
## around half the time of the original model
#################################

print(" ")
print("Training the reduced model:")

tic = time.time()

model_trained = training_loop_with_stop(
    n_epochs = 5000, 
    learning_rate = 0.01,
    model = radnet_red,
    params = list(radnet_red.parameters()),
    x_train = x_med,
    y_train = y_med,
    stopping_value=0.01,
    verbose = True)

toc = time.time()
print()
print("Elapsed time:", toc - tic)
