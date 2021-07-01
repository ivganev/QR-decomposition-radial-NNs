import numpy as np
from typing import List
import time

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
## Run experiments from Section 6.3: Demonstrate compressed
##      network reaches same loss in less  time.
#################################
def run_experiments(args):
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
    
    if args.verbose:
        print("\nTraining the original model:")

    tic = time.time()

    model_trained = training_loop_with_stop(
        n_epochs = args.epochs, 
        learning_rate = 0.01,
        model = radnet,
        params = list(radnet.parameters()),
        x_train = x_med,
        y_train = y_med,
        stopping_value=0.01,
        verbose=args.verbose)

    toc = time.time()
    model_elapsed = toc - tic
    if args.verbose:
        print("\nElapsed time:", model_elapsed)


    #################################
    ## Run the reduced model until the loss is below 0.01
    ## This usually takes less than 15 seconds,
    ## around half the time of the original model
    #################################

    if args.verbose:
        print("\nTraining the reduced model:")

    tic = time.time()

    model_trained = training_loop_with_stop(
        n_epochs = args.epochs, 
        learning_rate = 0.01,
        model = radnet_red,
        params = list(radnet_red.parameters()),
        x_train = x_med,
        y_train = y_med,
        stopping_value=0.01,
        verbose = args.verbose)

    toc = time.time()
    red_model_elapsed = toc - tic
    if args.verbose:
        print("\nElapsed time:", red_model_elapsed)

    return model_elapsed, red_model_elapsed


def main():
    parser = argparse.ArgumentParser(description='Run experiments 6.1 and 6.2.')
    parser.add_argument('--trials', '-n', type=int, help='number of trials',
                    default=10)
    parser.add_argument('--epochs', '-e', type=int, help='number of epochs',
                    default=5000)
    parser.add_argument('--verbose', '-v', action='store_true',
                    help='print each output', default=False)
    args = parser.parse_args()

    elapsed = []
    elapsed_red = []
    print(f"Running Experiment 6.3 for {args.trials} trials.")
    for trial in tqdm(range(args.trials)):
        model_elapsed, red_model_elapsed = run_experiments(args)
        elapsed.append(model_elapsed)
        elapsed_red.append(red_model_elapsed)

    elapsed = torch.tensor(elapsed)
    elapsed_red = torch.tensor(elapsed_red)

    print("Experiment 6.3.  {0} Trials".format(args.trials))
    print("Full Model Training Time = {0:.3e} +/- {1:.3e}".format(torch.mean(elapsed),torch.std(elapsed)))
    print("Reduced Model Training Time = {0:.3e} +/- {1:.3e}".format(torch.mean(elapsed_red),torch.std(elapsed_red)))
    print("Full model takes {0:.3e} +/- {1:.3e} times longer to train".format(torch.mean(elapsed/elapsed_red),torch.std(elapsed/elapsed_red)))


main()
