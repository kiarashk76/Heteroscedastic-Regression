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
from ExpMultiDim_single_y import *
import argparse
from animate import animate
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--run_num', default="0")

    args = parser.parse_args()
    # run_num_list = list(range(int(args.start_run), int(args.end_run)))
    num_agent = 6

    params = {
        # experiment configs
        'num_runs': 1,
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
        'data_dim': 10,
        'hidden_layers_error': [[64, 64]] * 10,
        'batch_sizes': [128] * num_agent,
        'step_sizes': [2**-i for i in range(5, 16, 2)],
        'plot_colors': ['b'] * num_agent,
        'loss_type': ['2'] * num_agent,
        'bias_available': [True] * num_agent,
        'mu_training': [True] * 10,
    }
    params['hidden_layers_var'] = [[64, 64]] * 10


    # params['loss_type'] = ['2'] * 10
    # params['run_number'] = int(args.run_num)
    # exp_name = 'MD_Irre_linearNoise'
    # exp = MD_experiment_irreducible_linear(params, exp_name)
    # exp.run_experiment()

    params['loss_type'] = ['3'] * 10
    params['run_number'] = int(args.run_num)
    exp_name = 'MD_Irre_linearNoise_MSE'
    exp = MD_experiment_irreducible_linear(params, exp_name)
    exp.run_experiment()

#####################################################################################

#     params['loss_type'] = ['2'] * 10
#     params['run_number'] = int(args.run_num)
#     exp_name = 'MD_Bias_rangelinearBias'
#     exp = MD_experiment_rangeLBias(params, exp_name)
#     exp.run_experiment()
#     del exp
#
#     params['loss_type'] = ['3'] * 10
#     params['run_number'] = int(args.run_num)
#     exp_name = 'MD_Bias_rangelinearBias_MSE'
#     exp = MD_experiment_rangeLBias(params, exp_name)
#     exp.run_experiment()
#     del exp
#
# # ********#####################################################################################
#     params['loss_type'] = ['2'] * 10
#     params['run_number'] = int(args.run_num)
#     exp_name = 'MD_Irre_rangeUniformNoise'
#     exp = MD_experiment_irreducible_error(params, exp_name)
#     exp.run_experiment()
#     del exp
#
#     params['loss_type'] = ['3'] * 10
#     params['run_number'] = int(args.run_num)
#     exp_name = 'MD_Irre_rangeUniformNoise_MSE'
#     exp = MD_experiment_irreducible_error(params, exp_name)
#     exp.run_experiment()
#     del exp
# # ********#######################################################################################################################
# # relu activation
#     params['loss_type'] = ['2'] * 10
#     params['run_number'] = int(args.run_num)
#     exp_name = 'MD_Bias_quadraticBias1'
#     exp = MD_experiment_quadraticBias(params, exp_name)
#     exp.A = 2
#     exp.run_experiment()
#     del exp
#
#     params['loss_type'] = ['3'] * 10
#     params['run_number'] = int(args.run_num)
#     exp_name = 'MD_Bias_quadraticBias1_MSE'
#     exp = MD_experiment_quadraticBias(params, exp_name)
#     exp.A = 2
#     exp.run_experiment()
#     del exp
#
#
########################################################################################################################
#
#
#
#     params['loss_type'] = ['2'] * 10
#     params['run_number'] = int(args.run_num)
#     exp_name = 'MD_experiment_irreducible_linear_single_y'
#     exp = MD_experiment_irreducible_linear_single_y(params, exp_name)
#     exp.run_experiment()
#     del exp
#
#     params['loss_type'] = ['3'] * 10
#     params['run_number'] = int(args.run_num)
#     exp_name = 'MD_experiment_irreducible_linear_single_y_MSE'
#     exp = MD_experiment_irreducible_linear_single_y(params, exp_name)
#     exp.run_experiment()
#     del exp
#
#
#
#########################################################################################################################################
#
#
#
#     params['loss_type'] = ['2'] * 10
#     params['run_number'] = int(args.run_num)
#     exp_name = 'MD_experiment_rangeLBias_single_y'
#     exp = MD_experiment_rangeLBias_single_y(params, exp_name)
#     exp.run_experiment()
#     del exp
#
#     params['loss_type'] = ['3'] * 10
#     params['run_number'] = int(args.run_num)
#     exp_name = 'MD_experiment_rangeLBias_single_y_MSE'
#     exp = MD_experiment_rangeLBias_single_y(params, exp_name)
#     exp.run_experiment()
#     del exp
#
#
#########################################################################################################################################
    #
    #
    #
    # params['loss_type'] = ['2'] * 10
    # params['run_number'] = int(args.run_num)
    # exp_name = 'MD_experiment_irreducible_error_single_y'
    # exp = MD_experiment_irreducible_error_single_y(params, exp_name)
    # exp.run_experiment()
    # del exp
    #
    # params['loss_type'] = ['3'] * 10
    # params['run_number'] = int(args.run_num)
    # exp_name = 'MD_experiment_irreducible_error_single_y_MSE'
    # exp = MD_experiment_irreducible_error_single_y(params, exp_name)
    # exp.run_experiment()
    # del exp
    #
    #
    ########################################################################################################################
    #
    #
    #
    # params['loss_type'] = ['2'] * 10
    # params['run_number'] = int(args.run_num)
    # exp_name = 'MD_experiment_quadraticBias_single_y'
    # exp = MD_experiment_quadraticBias_single_y(params, exp_name)
    # exp.run_experiment()
    # del exp
    #
    # params['loss_type'] = ['3'] * 10
    # params['run_number'] = int(args.run_num)
    # exp_name = 'MD_experiment_quadraticBias_single_y_MSE'
    # exp = MD_experiment_quadraticBias_single_y(params, exp_name)
    # exp.run_experiment()
    # del exp


