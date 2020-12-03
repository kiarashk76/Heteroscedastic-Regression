import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from GeneralModel import GeneralModel
from time import time
import pickle

class experiment():
    def __init__(self, params, experiment_name):
        #experiment configs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.parameters = params
        self.num_runs = params['num_runs']
        self.num_epochs = params['num_epochs']
        self.num_data_points = params['num_data_points']
        self.plt_show = params['plt_show']
        self.plt_save = params['plt_save']
        self.plot_show_epoch_freq = params['plot_show_epoch_freq']

        #agents configs
        self.num_agents = params['num_agents']
        self.names = params['names']
        self.hidden_layers_mu = params['hidden_layers_mu']
        self.hidden_layers_var = params['hidden_layers_var']
        self.data_dim = params['data_dim']
        self.batch_sizes = params['batch_sizes']
        self.step_sizes = params['step_sizes']
        self.plot_colors = params['plot_colors']
        self.bias_available = params['bias_available']
        self.mu_training = params['mu_training']
        self.loss_type = params['loss_type']
        self.experiment_name = experiment_name

        self.error_list = np.zeros([self.num_runs, self.num_epochs, self.num_agents])
        self.error_list_sigma = np.zeros([self.num_runs, self.num_epochs, self.num_agents])
        self.learn_mu = np.zeros([self.num_runs, self.num_epochs, self.num_agents, self.num_data_points, self.data_dim])
        self.learn_var = np.zeros([self.num_runs, self.num_epochs, self.num_agents, self.num_data_points, self.data_dim])
    def create_dataset(self):
        # create the dataset
        range_data_points = (-1, 2)
        x = np.random.uniform(range_data_points[0], range_data_points[1], 
                              self.num_data_points * self.data_dim)
        x = np.reshape(np.sort(x), (self.num_data_points, self.data_dim))
        y = x + np.sin(4 * x) + np.sin(13 * x)
        mu = x + np.sin(4 * x) + np.sin(13 * x)
        self.x, self.y, self.mu = x.to(self.device), y.to(self.device), mu.to(self.device)

    def init_models(self):
        #initializing the models
        self.models = []
        for i in range(self.num_agents):
            params = {"hidden_layers_mu": self.hidden_layers_mu[i],
                      "hidden_layers_var": self.hidden_layers_var[i],
                      "data_dim": self.data_dim,
                      "batch_size": self.batch_sizes[i],
                      "step_size": self.step_sizes[i],
                      "name": self.names[i],
                      "bias_available":self.bias_available[i],
                      "loss_type":self.loss_type[i],
                      "device":self.device}
            m = GeneralModel(params)
            self.models.append(m)

    def run_experiment(self):
        for r in range(self.num_runs):
            np.random.seed(r)
            print('\n run number: ', r+1)

            self.create_dataset()
            self.init_models()
            #train
            for e in range(self.num_epochs):
                # train models
                self.train_models()
                self.validate_models(r, e)

        # error plot for mu
        for i in range(self.num_agents):
            err = np.mean(self.error_list, axis=0)[:, i]
            err_bar = np.std(self.error_list, axis=0)[:, i]
            self.drawPlotUncertainty(range(len(err)), err, err_bar, 'model '+ self.models[i].name, self.plot_colors[i])
        if self.plt_show:
            plt.title("error plot over all the runs")
            plt.legend()
            plt.show()
            plt.close()
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
            plt.title("error plot over all the runs for sigma")
            plt.legend()
            plt.show()
            plt.close()
        if self.plt_save:
            plt.title("error plot over all the runs for sigma")
            plt.savefig('plots/' + self.experiment_name + '@' + str(time()) + '.png')

        #average learning plot
        for i in range(self.num_agents):
            average_learn_mu = np.mean(self.learn_mu, axis=0)[:, i]
            average_learn_var = np.mean(self.learn_var, axis=0)[:, i]
            plt.plot(average_learn_var[-1], label='var')

        if self.plt_show:
            plt.title("average learning curve")
            plt.legend()
            plt.show()
            plt.close()

        with open('data/'+self.experiment_name+'.p', 'wb') as f:
            data = {
                'x': self.x,
                'y': self.y,
                'learn_mu': self.learn_mu,
                'learn_var': self.learn_var,
                'mu_error_list': self.error_list,
                'sigma_error_list': self.error_list_sigma,
                'params': self.parameters
            }
            pickle.dump(data, f)

    def train_models(self):
        for a, model in enumerate(self.models):
            mask = list(range(self.num_data_points))
            np.random.shuffle(mask)
            for i in range(self.num_data_points // self.batch_sizes[a]):
                # ind = np.random.choice(self.num_data_points, self.batch_sizes[a])
                ind = mask[i*self.batch_sizes[a]:(i+1)*self.batch_sizes[a]]
                batch_x, batch_y, batch_mu = self.x[ind], self.y[ind], self.mu[ind]
                # give batch_mu so mu not being learned
                if self.mu_training[a]:
                    model.train_model(batch_x, batch_y, batch_mu=None)
                else:
                    model.train_model(batch_x, batch_y, batch_mu=batch_mu)
    
    def validate_models2(self, run_number, epoch_number):
        #validate models
        for a, model in enumerate(self.models):
            mu, var = model.test_model(self.x, self.y)
            if not self.mu_training[a]:
                mu = torch.from_numpy(self.mu).float()
            self.learn_mu[run_number, epoch_number, a] = mu
            self.learn_var[run_number, epoch_number, a] = var
            distance = torch.dist(torch.from_numpy(self.y).float(), mu)
            self.error_list[run_number, epoch_number, a] = distance

            # draw plot till now
            if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
                # mu, var = model.test_model(x, y)
                # error = model.test_error(self.x, self.y) #only for GeneralModelwithError
                self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], var[:, 0], 'model '+ model.name, self.plot_colors[a])
        
        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
            plt.plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
            plt.title('models after '+ str(epoch_number) +' epochs in run number ' + str(run_number+1))
            plt.show()
            plt.close()
        
        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_save:
            plt.plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
            plt.title('models after '+ str(epoch_number) +' epochs in run number ' + str(run_number+1))
            plt.savefig('plots/' + self.experiment_name + f'{epoch_number:04}' + '.png')
            plt.close()

    def validate_models(self, run_number, epoch_number):
        # validate models
        ground_truth = False

        for a, model in enumerate(self.models):
            mu, var = model.test_model(self.x, self.y)
            if not self.mu_training[a]:
                mu = torch.from_numpy(self.mu).float()
            distance = torch.dist(torch.from_numpy(self.y).float(), mu)
            noise = (torch.from_numpy(self.y).float() - mu) ** 2

            sigma_distance = torch.dist(noise, var)
            self.error_list[run_number, epoch_number, a] = distance
            self.error_list_sigma[run_number, epoch_number, a] = sigma_distance
            self.learn_mu[run_number, epoch_number, a] = mu
            self.learn_var[run_number, epoch_number, a] = var
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
                axs[1].plot(self.x[:, 0], var[:, 0])

        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
            plt.show()
            plt.close()

        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_save:
            plt.plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
            plt.title('models after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))
            plt.savefig('plots/' + self.experiment_name + f'{epoch_number:04}' + '.png')
            plt.close()

    def drawPlotUncertainty(self, x, y, y_err, label, color, axis=plt):
        axis.plot(x, y, color, label=label)
        axis.fill_between(x,
                        y - y_err,
                        y + y_err,
                        facecolor=color, alpha=0.4, edgecolor='none')
