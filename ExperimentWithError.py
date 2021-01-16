import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Experiment import experiment
from GeneralModel import GeneralModelwithError
import pickle
from tqdm import tqdm
from time import time

class experimentWithError(experiment):
    def __init__(self, params, experiment_name):
        experiment.__init__(self, params, experiment_name)
        self.hidden_layers_error = params['hidden_layers_error']

    def init_models(self):
        #initializing the models
        self.models = []
        for i in range(self.num_agents):
            params = {"hidden_layers_mu": self.hidden_layers_mu[i],
                      "data_dim": self.data_dim,
                      "hidden_layers_error": self.hidden_layers_error[i],
                      "batch_size": self.batch_sizes[i],
                      "step_size": self.step_sizes[i],
                      "name": self.names[i],
                      "bias_available": self.bias_available[i],
                      "loss_type": self.loss_type[i]}
            m = GeneralModelwithError(params)
            self.models.append(m)

    def run_experiment(self):
        np.random.seed(0)
        self.create_dataset()
        for r in tqdm(range(self.num_runs)):
            np.random.seed(r)
            # print('\n run number: ', r+1)

            self.init_models()
            # train
            for e in tqdm(range(self.num_epochs)):
                # train models
                self.train_models()
                self.validate_models(r, e)

        # error plot for mu
        for i in range(self.num_agents):
            err = np.mean(self.error_list, axis=0)[:, i]
            err_bar = np.std(self.error_list, axis=0)[:, i]
            self.drawPlotUncertainty(range(len(err)), err, err_bar, 'model ' + self.models[i].name, self.plot_colors[i])
        if self.plt_show:
            pass
            # plt.title("error plot over all the runs")
            # plt.legend()
            # plt.show()
            # plt.close()
        if self.plt_save:
            plt.title("error plot over all the runs")
            plt.savefig('plots/' + self.experiment_name + '@' + str(time()) + '.png')

        # error plot for sigma
        for i in range(self.num_agents):
            err = np.mean(self.error_list_sigma, axis=0)[:, i]
            err_bar = np.std(self.error_list_sigma, axis=0)[:, i]
            self.drawPlotUncertainty(range(len(err)), err, err_bar, 'model ' + self.models[i].name,
                                     self.plot_colors[i])
        if self.plt_show:
            pass
            # plt.title("error plot over all the runs for sigma")
            # plt.legend()
            # plt.show()
            # plt.close()
        if self.plt_save:
            plt.title("error plot over all the runs for sigma")
            plt.savefig('plots/' + self.experiment_name + '@' + str(time()) + '.png')

        # average learning plot
        for i in range(self.num_agents):
            average_learn_mu = np.mean(self.learn_mu, axis=0)[:, i]
            average_learn_var = np.mean(self.learn_var, axis=0)[:, i]
            plt.plot(average_learn_var[-1], label='var')

        if self.plt_show:
            pass
            # plt.title("average learning curve")
            # plt.legend()
            # plt.show()
            # plt.close()

        with open('data/expwitherror/' + self.experiment_name + '.p', 'wb') as f:
            data = {
                'x': self.x,
                'y': self.y,
                'avg_learn_mu': np.mean(self.learn_mu, axis=0),
                'avg_learn_var': np.mean(self.learn_var, axis=0),
                'last_mu': self.learn_mu[:, -10:-1],
                'last_var': self.learn_var[:, -10:-1],
                'mu_error_list': self.error_list,
                'sigma_error_list': self.error_list_sigma,
                'params': self.parameters
            }
            pickle.dump(data, f)
    
    
    def validate_models(self, run_number, epoch_number):
        #validate models
        # validate models
        ground_truth = False

        for a, model in enumerate(self.models):
            mu, _ = model.test_model(self.x, self.y)
            var = model.error_network.test_error(self.x, self.y)
            if not self.mu_training[a]:
                mu = torch.from_numpy(self.mu).float()
            distance = torch.dist(torch.from_numpy(self.y).float(), mu)
            noise = (torch.from_numpy(self.y).float() - mu) ** 2

            sigma_distance = torch.dist(noise, var)
            self.error_list[run_number, epoch_number, a] = distance
            self.error_list_sigma[run_number, epoch_number, a] = sigma_distance
            self.learn_mu[run_number, epoch_number, a] = mu.to(torch.device("cpu"))
            self.learn_var[run_number, epoch_number, a] = var.to(torch.device("cpu"))
            # draw plot till now
            if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
                # mu, var = model.test_model(x, y)
                # error = model.test_error(self.x, self.y) #only for GeneralModelwithError
                fig, axs = plt.subplots(2, 1)
                if not ground_truth:
                    axs[0].plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
                    axs[0].title.set_text(
                        'models after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))
                    axs[1].plot(self.x, noise, 'ko', markersize=0.5, label='ground truth',
                                alpha=0.5)  # '''0.5* self.x'''
                    axs[1].title.set_text(
                        'sigma after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))

                self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], var[:, 0], 'model ' + model.name,
                                         self.plot_colors[a],
                                         axs[0])
                # self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], torch.from_numpy(np.zeros_like(mu[:, 0])) , 'model ' + model.name,
                #                          self.plot_colors[a],
                #                          axs[0])
                axs[1].plot(self.x[:, 0], var[:, 0])

        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
            plt.show()
            plt.close()

        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_save:
            plt.plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
            plt.title('models after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))
            plt.savefig('plots/' + self.experiment_name + f'{epoch_number:04}' + '.png')
            plt.close()