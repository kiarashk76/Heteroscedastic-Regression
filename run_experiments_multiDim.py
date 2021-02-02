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

from ExpMultiDim import *

from animate import animate

if __name__ == "__main__":
    num_agent = 6

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
        'hidden_layers_var':[[]] * num_agent,
        'data_dim': 20,
        'hidden_layers_error': [[64, 64]] * 10,
        'batch_sizes': [128] * num_agent,
        'step_sizes': [2**-i for i in range(5, 16, 2)],
        'plot_colors': ['b'] * num_agent,
        'loss_type': ['2'] * num_agent,
        'bias_available': [True] * num_agent,
        'mu_training': [True] * 10,
    }
    params['loss_type'] = ['2'] * 10
    # exp_name = 'MD_Irre_linearNoise'
    # exp = MD_experiment_irreducible_linear(params, exp_name)
    # exp.run_experiment()

    # exp_name = 'MD_Bias_rangelinearBias'
    # exp = MD_experiment_rangeLBias(params, exp_name)
    # exp.run_experiment()

    # # ********
    # exp_name = 'MD_Irre_rangeUniformNoise'
    # exp = MD_experiment_irreducible_error(params, exp_name)
    # exp.run_experiment()

    # # ********
    # # relu activation
    # exp_name = 'MD_Bias_quadraticBias1'
    # exp = MD_experiment_quadraticBias(params, exp_name)
    # exp.A = 2
    # exp.run_experiment()

    exp_name = 'MD_experiment_irreducible_error_single_y'
    exp = MD_experiment_irreducible_error_single_y(params, exp_name)
    exp.run_experiment()


