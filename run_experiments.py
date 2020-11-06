#!/usr/bin/env python
#coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from Exp_Bias import experiment_bias, experiment_bias2
from Exp_IrreducibleError import experiment_irreducible_error
from Exp_withError import experimentWithError

if __name__ == "__main__":
    params = {
            #experiment configs
            'num_runs': 1,
            'num_epochs': 100,
            'num_data_points': 1000,
            'plt_show': True,
            'plot_show_epoch_freq': 10,
            
            #agents configs
            'num_agents': 1,
            'names': ['linear'],
            'hidden_layers_mu': [[]],
            'hidden_layers_var':[[]],
            'data_dim':2,
            'hidden_layers_error':[[]],
            'batch_sizes': [8],
            'step_sizes': [0.001],
            'plot_colors': ['r']          
        }

    exp_bias = experiment_bias(params)
    exp_bias.run_experiment()

    # exp_bias = experiment_bias2(params)
    # exp_bias.run_experiment()

    # exp_irriducible = experiment_irreducible_error(params)
    # exp_irriducible.run_experiment()

    # mu, sigma = exp_irriducible.models[0].test_model(exp_irriducible.x, ty)
    # # real_error = np.abs(exp.y - 2*exp.x)
    # # print(sigma.shape, real_error.shape)
    # # plt.plot(exp.x, real_error, label='real')
    # plt.plot(exp_irriducible.x, sigma, label='pred')
    # plt.legend()
    # plt.show()
    # print(exp_irriducible.x, sigma)