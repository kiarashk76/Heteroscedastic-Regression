import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from Exp_Homoscedastic import *


if __name__ == "__main__":
    num_agent = 1

    params = {
        # experiment configs
        'num_runs': 1,
        'num_epochs': 801,
        'num_data_points': 2000,
        'plt_show': True,
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
        'step_sizes': [2 ** -9],  # [2**-i for i in range(5, 16, 2)],
        'plot_colors': ['b'] * num_agent,
        'loss_type': ['1'] * num_agent,
        'bias_available': [True] * num_agent,
        'mu_training': [True] * num_agent,
    }


    # ********
    params['hidden_layers_var'] = [[64, 64]] * 10

    params['loss_type'] = ['1'] * 10
    # relu activation
    exp_name = 'Homo1'
    exp = experiment_homo1(params, exp_name)
    exp.run_experiment()

    params['loss_type'] = ['3'] * 10
    # relu activation
    exp_name = 'RegHomo1'
    exp = experiment_homo1(params, exp_name)
    exp.run_experiment()

    params['hidden_layers_mu'] = [[64, 64]] * 10
    params['loss_type'] = ['1'] * 10
    # relu activation
    exp_name = 'Homo2'
    exp = experiment_homo1(params, exp_name)
    exp.run_experiment()

    params['loss_type'] = ['3'] * 10
    # relu activation
    exp_name = 'RegHomo2'
    exp = experiment_homo1(params, exp_name)
    exp.run_experiment()