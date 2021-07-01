import numpy as np
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import argparse
from tqdm import tqdm

from source import *
torch.manual_seed(1)


#################################
## Run experiments from Section 6.1 and Section 6.2
## - Experiment Section 6.1: Verify Theorem 4.6, that
##     Neural functions f_W and f_R match
## - Experiment Section 6.2: Verify Theorem 5.5, that
##     projected gradient descent of f_W matches
##     gradient descent for f_R
#################################
def run_experiments(args):

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

    if args.verbose:
        print("\nTraining the original model with projected GD:")

    model_trained, model_losses = training_loop_proj_GD(
        n_epochs = args.epochs, 
        learning_rate = 0.01,
        model = radnet_trans,
        params = list(radnet_trans.parameters()),
        original_dimensions = radnet_trans.dims,
        reduced_dimensions = radnet_trans.dims_red,
        x_train = x_synth,
        y_train = y_synth,
        verbose=args.verbose)


    #################################
    ## Train the reduced model using ordinary gradient descent
    #################################
    
    if args.verbose:
        print("\nTraining the reduced model with ordinary GD:")

    model_red_trained, model_red_losses = training_loop(
        n_epochs = args.epochs, 
        learning_rate = 0.01,
        model = radnet_red,
        params = list(radnet_red.parameters()),
        x_train = x_synth,
        y_train = y_synth,
        verbose=args.verbose)

    return model_losses, model_red_losses


def main():
    parser = argparse.ArgumentParser(description='Run experiments 6.1 and 6.2.')
    parser.add_argument('--trials', '-n', type=int, help='number of trials',
                    default=10)
    parser.add_argument('--epochs', '-e', type=int, help='number of epochs',
                    default=3000)
    parser.add_argument('--verbose', '-v', action='store_true',
                    help='print each output', default=False)
    args = parser.parse_args()

    untrained_loss_model = []
    trained_loss_model = []
    untrained_loss_red_model = []
    trained_loss_red_model = []
    print(f"Running Experiments 6.1 and 6.2 for {args.trials} trials.")
    for trial in tqdm(range(args.trials)):
        model_losses, model_red_losses = run_experiments(args)
        untrained_loss_model.append(model_losses[0])
        trained_loss_model.append(model_losses[-1])
        untrained_loss_red_model.append(model_red_losses[0])
        trained_loss_red_model.append(model_red_losses[-1])

    untrained_loss_model = torch.tensor(untrained_loss_model)
    trained_loss_model = torch.tensor(trained_loss_model)
    untrained_loss_red_model = torch.tensor(untrained_loss_red_model)
    trained_loss_red_model = torch.tensor(trained_loss_red_model)

    error_6_1 = torch.abs(untrained_loss_model - untrained_loss_red_model)
    error_6_2 = torch.abs(trained_loss_model - trained_loss_red_model)

    print("Experiment 6.1.  Over {0} trials, Error = {1:.3e} +/- {2:.3e}".
        format(args.trials,torch.mean(error_6_1),torch.std(error_6_1)))
    if args.verbose:
        print("Errors:",error_6_1)
    print("Experiment 6.2.  Over {0} trials, Error = {1:.3e} +/- {2:.3e}".
        format(args.trials,torch.mean(error_6_2),torch.std(error_6_2)))
    if args.verbose:
        print("Errors:",error_6_2)

main()

