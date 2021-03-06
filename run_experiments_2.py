#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from Exp_Bias import *
from Exp_IrreducibleError import *
from ExperimentWithError import experimentWithError
from Experiment import experiment

import argparse

from animate import animate

if __name__ == "__main__":
    num_agent = 6
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default="8")
    args = parser.parse_args()

    params = {
        # experiment configs
        'num_runs': 30,
        'num_epochs': 2000,
        'num_data_points': 2000,
        'plt_show': False,
        'plt_save': False,
        'plot_show_epoch_freq': 100,

        # agents configs
        'num_agents': num_agent,
        'names': ['het'] * num_agent,
        'hidden_layers_mu': [[]] * num_agent,
        'hidden_layers_var': [[]] * num_agent,
        'data_dim': 1,
        'hidden_layers_error': [[]] * num_agent,
        'batch_sizes': [128] * num_agent,
        'step_sizes': [2**-i for i in range(5, 16, 2)],
        'plot_colors': ['r'] * num_agent,
        'loss_type': ['1'] * num_agent,
        'bias_available': [True] * num_agent,
        'mu_training': [True] * num_agent,
    }

    # *************
    params['batch_sizes'] = [128] * 10
    params['hidden_layers_mu'] = [[32,32]] * 10
    params['hidden_layers_var'] =  [[]] * 10
    params['loss_type'] =  ['1'] * 10



    experiment_to_run = args.exp_num

    if experiment_to_run == "1":
        exp_name = 'HetFail1_Irre_rangeUniformNoise1'
        exp = experiment_irreducible_error10(params, exp_name)
        exp.run_experiment()

    elif experiment_to_run == "2":
        params['hidden_layers_mu'] = [[24, 24]] * 10
        exp_name = 'HetFail1_Irre_rangeUniformNoise2'
        exp = experiment_irreducible_error10(params, exp_name)
        exp.run_experiment()

    elif experiment_to_run == "3":
        params['hidden_layers_mu'] = [[16, 16]] * 10
        exp_name = 'HetFail1_Irre_rangeUniformNoise3'
        exp = experiment_irreducible_error10(params, exp_name)
        exp.run_experiment()

    elif experiment_to_run == "4":
        params['hidden_layers_mu'] = [[12, 12]] * 10
        exp_name = 'HetFail1_Irre_rangeUniformNoise4'
        exp = experiment_irreducible_error10(params, exp_name)
        exp.run_experiment()

    elif experiment_to_run == "5":
        params['hidden_layers_mu'] = [[8, 8]] * 10
        exp_name = 'HetFail1_Irre_rangeUniformNoise5'
        exp = experiment_irreducible_error10(params, exp_name)
        exp.run_experiment()

    #******
    elif experiment_to_run == "6":
        params['loss_type'] =  ['3'] * 10
        params['hidden_layers_mu'] = [[32,32]] * 10
        exp_name = 'RegFail1_Irre_rangeUniformNoise1'
        exp = experiment_irreducible_error10(params, exp_name)
        exp.run_experiment()

    elif experiment_to_run == "7":
        params['hidden_layers_mu'] = [[24, 24]] * 10
        exp_name = 'RegFail1_Irre_rangeUniformNoise2'
        exp = experiment_irreducible_error10(params, exp_name)
        exp.run_experiment()

    elif experiment_to_run == "8":
        params['hidden_layers_mu'] = [[16, 16]] * 10
        exp_name = 'RegFail1_Irre_rangeUniformNoise3'
        exp = experiment_irreducible_error10(params, exp_name)
        exp.run_experiment()

    elif experiment_to_run == "9":
        params['hidden_layers_mu'] = [[12, 12]] * 10
        exp_name = 'RegFail1_Irre_rangeUniformNoise4'
        exp = experiment_irreducible_error10(params, exp_name)
        exp.run_experiment()

    elif experiment_to_run == "10":
        params['hidden_layers_mu'] = [[8, 8]] * 10
        exp_name = 'RegFail1_Irre_rangeUniformNoise5'
        exp = experiment_irreducible_error10(params, exp_name)
        exp.run_experiment()