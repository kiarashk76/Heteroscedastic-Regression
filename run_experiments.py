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

from Exp_Bias import *
from Exp_IrreducibleError import *
from Exp_withError import experimentWithError
from Experiment import experiment

from animate import animate

if __name__ == "__main__":
    params = {
            #experiment configs
            'num_runs': 30,
            'num_epochs': 500,
            'num_data_points': 5000,
            'plt_show': True,
            'plt_save': False,
            'plot_show_epoch_freq': 10,
            
            #agents configs
            'num_agents': 10,
            'names': ['het']*10,
            'hidden_layers_mu': [[]]*10,
            'hidden_layers_var':[[]]*10,
            'data_dim':1,
            'hidden_layers_error':[[]]*10,
            'batch_sizes': [16]*10,
            'step_sizes': [2**-i for i in range(5, 15)],
            'plot_colors': ['r']*10,
            'loss_type': ['1']*10,
            'bias_available':[True]*10,
            'mu_training':[False]*10,
        }

#  ******************          Fixed Mu Experiments !
    exp_name = 'Irre_fixedMu_linearNoise'
    exp = experiment_irreducible_error1(params, exp_name)
    exp.run_experiment()

    exp_name = 'IrreBias_fixedMu_linearNoise_fixedBias'
    exp = experiment_irreducible_error2(params, exp_name)
    exp.run_experiment()

    exp_name = 'Bias_fixedMu_linearBias'
    exp = experiment_bias1(params, exp_name)
    exp.run_experiment()

# ********
    # relu activation
    exp_name = 'Irre_fixedMu_rangeUniformNoise'
    params['hidden_layers_var'] = [[64, 64]] * 10
    exp = experiment_irreducible_error4(params, exp_name)
    exp.run_experiment()

    #relu activation
    params['hidden_layers_var'] = [[64]]*10
    exp_name = 'Bias_fixedMu_rangefixedBias'
    exp = experiment_bias2(params, exp_name)
    exp.run_experiment()

    # tanh activation
    params['hidden_layers_var'] = [[64, 64]]*10
    exp_name = 'IrreBias_fixedMu_sinNoise_fixedBias'
    exp = experiment_irreducible_error3(params, exp_name)
    exp.run_experiment()


#  ******************      Not Fixed Mu Experiments !
# ********
    #relu activation
    params['hidden_layers_var'] = [[64]]*10
    params['mu_training'] = [True]*10
    exp_name = 'Bias_rangefixedBias'
    exp = experiment_bias2(params, exp_name)
    exp.run_experiment()

    params['hidden_layers_var'] = [[64]] * 10
    params['mu_training'] = [True] * 10
    params['loss_type'] =  ['3']*10
    exp_name = 'RegularReg_Bias_rangefixedBias'
    exp = experiment_bias2(params, exp_name)
    exp.run_experiment()

# ********
    # relu activation
    exp_name = 'Irre_rangeUniformNoise'
    params['hidden_layers_var'] = [[64, 64]] * 10
    params['mu_training'] = [True]*10
    exp = experiment_irreducible_error4(params, exp_name)
    exp.run_experiment()

    exp_name = 'RegularReg_Irre_rangeUniformNoise'
    params['hidden_layers_var'] = [[64, 64]] * 10
    params['mu_training'] = [True] * 10
    params['loss_type'] = ['3']*10
    exp = experiment_irreducible_error4(params, exp_name)
    exp.run_experiment()

# ********
    # relu activation
    params['hidden_layers_var'] = [[64, 64]] * 10
    params['mu_training'] = [True] * 10
    params['loss_type'] =  ['1']*10

    exp_name = 'Bias_quadraticBias1'
    exp = experiment_bias3(params, exp_name)
    exp.A = 2
    exp.run_experiment()

    exp_name = 'Bias_quadraticBias2'
    exp = experiment_bias3(params, exp_name)
    exp.A = 0.5
    exp.run_experiment()

    exp_name = 'Bias_quadraticBias3'
    exp = experiment_bias3(params, exp_name)
    exp.A = 2**-4
    exp.run_experiment()

    params['hidden_layers_var'] = [[64, 64]] * 10
    params['mu_training'] = [True] * 10
    params['loss_type'] = ['3'] * 10

    exp_name = 'RegularReg_Bias_quadraticBias1'
    exp = experiment_bias3(params, exp_name)
    exp.A = 2
    exp.run_experiment()

    exp_name = 'RegularReg_Bias_quadraticBias2'
    exp = experiment_bias3(params, exp_name)
    exp.A = 0.5
    exp.run_experiment()

    exp_name = 'RegularReg_Bias_quadraticBias3'
    exp = experiment_bias3(params, exp_name)
    exp.A = 2 ** -4
    exp.run_experiment()
# ********

    # animate
    fp_in = "plots/" + exp_name + "/*.png"
    fp_out = "plots/" + exp_name + "/animate.gif"
    animate(fp_in, fp_out)
