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
from Exp_withError import experimentWithError
from Experiment import experiment

from animate import animate

if __name__ == "__main__":
    num_agent = 1

    params = {
        # experiment configs
        'num_runs': 30,
        'num_epochs': 800,
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
        'step_sizes': [0.001],  # [2**-i for i in range(5, 16, 2)],
        'plot_colors': ['r'] * num_agent,
        'loss_type': ['1'] * num_agent,
        'bias_available': [True] * num_agent,
        'mu_training': [True] * num_agent,
    }

    params['mu_training'] = [True] * num_agent
    params['hidden_layers_var'] = [[]] * num_agent
    params['loss_type'] = ['3'] * num_agent

    # ********
    # relu activation
    params['hidden_layers_var'] = [[64, 64]] * 10
    params['loss_type'] = ['1'] * 10

    exp_name = 'Bias_quadraticBias'
    exp = experiment_bias3(params, exp_name)
    exp.A = 0.5
    exp.run_experiment()

    params['hidden_layers_var'] = [[64, 64]] * 10
    params['loss_type'] = ['3'] * 10

    exp_name = 'RegularReg_Bias_quadraticBias'
    exp = experiment_bias3(params, exp_name)
    exp.A = 0.5
    exp.run_experiment()



