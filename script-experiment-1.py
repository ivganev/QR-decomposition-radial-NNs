import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

from source import *



#################################
## Small synthetic dataset based on the normal distribution.
#################################

x_synth = torch.tensor([-3.0 + i/20 for i in range(121)]).unsqueeze(1)
y_synth = torch.zeros(x_synth.shape)
for i in range(x_synth.shape[0]):
    y_synth[i] = torch.exp(-(x_synth[i])**2)


#################################
## Instantiate a radial neural network
#################################

radnet = RadNet(eta=torch.sigmoid, dims=[1,6,7,1], has_bias=False)
orig_dims = radnet.dims
reduce_dims = radnet.dims_red


#################################
## Extract Q_inv acting on U for future reference
#################################

exported_weights = radnet.export_weights()
exported_weights.transformed_representation()
Q_inv_U_orig = exported_weights.Q_inv_U


#################################
## Compute the transformed network (Q_inv acting on W) 
## and the reduced network (R)
#################################

radnet_trans = radnet.transformed_network()
radnet_red = radnet.reduced_network()


#################################
## Check that the reduced and  original network
## have the same loss function on the synthetic data set. 
#################################

assert all([all(radnet(x_synth) - radnet_red(x_synth) < 0.0001), \
(loss_fn(radnet(x_synth), y_synth) - loss_fn(radnet_red(x_synth), y_synth) < 0.0001).item()]), \
"Loss functions do not match"


#################################
## Train the transformed model using projected gradient descent
#################################

print(" ")
print("Training the original model with projected GD:")

model_trained = training_loop_proj_GD(
    n_epochs = 3000, 
    learning_rate = 0.01,
    model = radnet_trans,
    params = list(radnet_trans.parameters()),
    original_dimensions = radnet_trans.dims,
    reduced_dimensions = radnet_trans.dims_red,
    x_train = x_synth,
    y_train = y_synth)


#################################
## Train the reduced model using ordinary gradient descent
#################################

print(" ")
print("Training the reduced model with ordinary GD:")

model_red_trained = training_loop(
    n_epochs = 3000, 
    learning_rate = 0.01,
    model = radnet_red,
    params = list(radnet_red.parameters()),
    x_train = x_synth,
    y_train = y_synth)

