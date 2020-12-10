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
    num_agent = 1

    params = {
            #experiment configs
            'num_runs': 1,
            'num_epochs': 801,
            'num_data_points':2000,
            'plt_show': True,
            'plt_save': False,
            'plot_show_epoch_freq': 100,
            
            #agents configs
            'num_agents': num_agent,
            'names': ['het'] * num_agent,
            'hidden_layers_mu': [[]] * num_agent,
            'hidden_layers_var': [[]] * num_agent,
            'data_dim':1,
            'hidden_layers_error': [[]] * num_agent,
            'batch_sizes': [128] * num_agent,
            'step_sizes': [2**-9],#[2**-i for i in range(5, 16, 2)],
            'plot_colors': ['b'] * num_agent,
            'loss_type': ['1'] * num_agent,
            'bias_available': [True] * num_agent,
            'mu_training': [True] * num_agent,
        }





#  ******************          Fixed Mu Experiments !
    params['mu_training'] = [False] * num_agent
    params['hidden_layers_var'] = [[]] * num_agent
    params['loss_type'] = ['1'] * num_agent

    exp_name = 'Irre_fixedMu_linearNoise'
    exp = experiment_irreducible_error1(params, exp_name)
    # exp.run_experiment()

    exp_name = 'IrreBias_fixedMu_linearNoise_fixedBias'
    exp = experiment_irreducible_error2(params, exp_name)
    # exp.run_experiment()

    params['hidden_layers_var'] = [[64,64]] * 10
    exp_name = 'Bias_fixedMu_linearBias'
    exp = experiment_bias1(params, exp_name)
    # exp.run_experiment()

# ********
    params['hidden_layers_var'] = [[64, 64]] * 10

    # relu activation
    exp_name = 'Irre_fixedMu_rangeUniformNoise'
    exp = experiment_irreducible_error4(params, exp_name)
    # exp.run_experiment()

    #relu activation
    exp_name = 'Bias_fixedMu_rangefixedBias'
    exp = experiment_bias2(params, exp_name)
    # exp.run_experiment()

    # relu activation
    exp_name = 'Bias_fixedMu_rangelinearBias'
    exp = experiment_bias4(params, exp_name)
    # exp.run_experiment()

    # tanh activation
    exp_name = 'IrreBias_fixedMu_sinNoise_fixedBias'
    exp = experiment_irreducible_error3(params, exp_name)
    # exp.run_experiment()


#  ******************      Not Fixed Mu Experiments !
    params['mu_training'] = [True]*10

# ********
    params['hidden_layers_var'] = [[64, 64]]*10

    params['loss_type'] = ['1']*10
    #relu activation
    exp_name = 'Bias_rangefixedBias'
    exp = experiment_bias2(params, exp_name)
    # exp.run_experiment()

    params['loss_type'] = ['3']*10
    exp_name = 'RegularReg_Bias_rangefixedBias'
    exp = experiment_bias2(params, exp_name)
    # exp.run_experiment()

# ********
    params['hidden_layers_var'] = [[64, 64]] * 10

    params['loss_type'] = ['1'] * 10
    # relu activation
    exp_name = 'Bias_rangelinearBias'
    exp = experiment_bias4(params, exp_name)
    # exp.run_experiment()

    params['loss_type'] = ['3'] * 10
    exp_name = 'RegularReg_Bias_rangelinearBias'
    exp = experiment_bias4(params, exp_name)
    # exp.run_experiment()

# ********
    params['hidden_layers_var'] = [[64, 64]] * 10

    # relu activation
    params['loss_type'] = ['1'] * 10
    exp_name = 'Irre_rangeUniformNoise'
    exp = experiment_irreducible_error4(params, exp_name)
    # exp.run_experiment()

    params['loss_type'] = ['3'] * 10
    exp_name = 'RegularReg_Irre_rangeUniformNoise'
    exp = experiment_irreducible_error4(params, exp_name)
    # exp.run_experiment()

# ********
    # relu activation
    params['hidden_layers_var'] = [[64, 64]] * 10
    params['loss_type'] =  ['1']*10

    exp_name = 'Bias_quadraticBias1'
    exp = experiment_bias3(params, exp_name)
    exp.A = 2
    # exp.run_experiment()

    exp_name = 'Bias_quadraticBias2'
    exp = experiment_bias3(params, exp_name)
    exp.A = 0.5
    # exp.run_experiment()

    exp_name = 'Bias_quadraticBias3'
    exp = experiment_bias3(params, exp_name)
    exp.A = 2**-4
    # exp.run_experiment()

    params['hidden_layers_var'] = [[64, 64]] * 10
    params['loss_type'] = ['3'] * 10

    exp_name = 'RegularReg_Bias_quadraticBias1'
    exp = experiment_bias3(params, exp_name)
    exp.A = 2
    # exp.run_experiment()

    exp_name = 'RegularReg_Bias_quadraticBias2'
    exp = experiment_bias3(params, exp_name)
    exp.A = 0.5
    # exp.run_experiment()

    exp_name = 'RegularReg_Bias_quadraticBias3'
    exp = experiment_bias3(params, exp_name)
    exp.A = 2 ** -4
    # exp.run_experiment()
# ********

    # relu activation
    params['hidden_layers_var'] = [[64, 64]] * 10
    params['loss_type'] = ['1'] * 10

    exp_name = 'Bias_FancyquadraticBias1'
    exp = experiment_bias5(params, exp_name)
    exp.A = 2
    # exp.run_experiment()

    exp_name = 'Bias_FancyquadraticBias2'
    exp = experiment_bias5(params, exp_name)
    exp.A = 0.5
    # exp.run_experiment()

    exp_name = 'Bias_FancyquadraticBias3'
    exp = experiment_bias5(params, exp_name)
    exp.A = 2 ** -4
    # exp.run_experiment()

    params['hidden_layers_var'] = [[64, 64]] * 10
    params['loss_type'] = ['3'] * 10

    exp_name = 'RegularReg_Bias_FancyquadraticBias1'
    exp = experiment_bias5(params, exp_name)
    exp.A = 2
    # exp.run_experiment()

    exp_name = 'RegularReg_Bias_FancyquadraticBias2'
    exp = experiment_bias5(params, exp_name)
    exp.A = 0.5
    # exp.run_experiment()

    exp_name = 'RegularReg_Bias_FancyquadraticBias3'
    exp = experiment_bias5(params, exp_name)
    exp.A = 2 ** -4
    # exp.run_experiment()

# **********
    params['batch_sizes'] = [128]*10
    params['hidden_layers_mu'] = [[32,32]] * 10
    params['hidden_layers_var'] =  [[]] * 10
    params['loss_type'] =  ['1'] * 10

    exp_name = 'HetFail1_Irre_rangeUniformNoise1'
    exp = experiment_irreducible_error5(params, exp_name)
    # exp.run_experiment()

    params['hidden_layers_mu'] = [[24, 24]] * 10
    exp_name = 'HetFail1_Irre_rangeUniformNoise2'
    exp = experiment_irreducible_error5(params, exp_name)
    # exp.run_experiment()


    params['hidden_layers_mu'] = [[16, 16]] * 10
    exp_name = 'HetFail1_Irre_rangeUniformNoise3'
    exp = experiment_irreducible_error5(params, exp_name)
    # exp.run_experiment()

    params['hidden_layers_mu'] = [[12, 12]] * 10
    exp_name = 'HetFail1_Irre_rangeUniformNoise4'
    exp = experiment_irreducible_error5(params, exp_name)
    # exp.run_experiment()

    params['hidden_layers_mu'] = [[8, 8]] * 10
    params['hidden_layers_var'] = [[]] * 10

    exp_name = 'HetFail1_Irre_rangeUniformNoise5'
    exp = experiment_irreducible_error5(params, exp_name)
    # exp.run_experiment()


    #******
    params['loss_type'] =  ['3'] * 10
    params['hidden_layers_mu'] = [[32,32]] * 10
    exp_name = 'RegFail1_Irre_rangeUniformNoise1'
    exp = experiment_irreducible_error5(params, exp_name)
    # exp.run_experiment()

    params['hidden_layers_mu'] = [[24, 24]] * 10
    exp_name = 'RegFail1_Irre_rangeUniformNoise2'
    exp = experiment_irreducible_error5(params, exp_name)
    # exp.run_experiment()


    params['hidden_layers_mu'] = [[16, 16]] * 10
    exp_name = 'RegFail1_Irre_rangeUniformNoise3'
    exp = experiment_irreducible_error5(params, exp_name)
    # exp.run_experiment()

    params['hidden_layers_mu'] = [[12, 12]] * 10
    exp_name = 'RegFail1_Irre_rangeUniformNoise4'
    exp = experiment_irreducible_error5(params, exp_name)
    # exp.run_experiment()

    params['hidden_layers_mu'] = [[8, 8]] * 10

    exp_name = 'RegFail1_Irre_rangeUniformNoise5'
    exp = experiment_irreducible_error5(params, exp_name)
    # exp.run_experiment()

    # animate
    fp_in = "plots/" + exp_name + "/*.png"
    fp_out = "plots/" + exp_name + "/animate.gif"
    animate(fp_in, fp_out)
