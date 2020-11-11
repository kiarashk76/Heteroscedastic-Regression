import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from GeneralModel import GeneralModel
from time import time

class experiment():
    def __init__(self, params):
        #experiment configs
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

        self.error_list = np.zeros([self.num_runs, self.num_epochs, self.num_agents])
    def create_dataset(self):
        # create the dataset
        range_data_points = (-1, 2)
        x = np.random.uniform(range_data_points[0], range_data_points[1], 
                              self.num_data_points * self.data_dim)
        x = np.reshape(np.sort(x), (self.num_data_points, self.data_dim))
        y = x + np.sin(4 * x) + np.sin(13 * x)
        self.x, self.y = x, y

    def init_models(self):
        #initializing the models
        self.models = []
        for i in range(self.num_agents):
            params = {"hidden_layers_mu": self.hidden_layers_mu[i],
                      "hidden_layers_var": self.hidden_layers_var[i],
                      "data_dim": self.data_dim,
                      "batch_size": self.batch_sizes[i],
                      "step_size": self.step_sizes[i],
                      "name": self.names[i]}
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

        # error plot
        for i in range(self.num_agents):
            err = np.mean(self.error_list, axis=0)[:, i]
            err_bar = np.std(self.error_list, axis=0)[:, i]
            self.drawPlotUncertainty(range(len(err)), err, err_bar, 'model '+ self.models[i].name, self.plot_colors[i])
        if self.plt_save:
            plt.title("error plot over all the runs")
            plt.savefig('plots/new' + str(time()) + '.png')
        
    
    def train_models(self):
        for a, model in enumerate(self.models):
            for _ in range(self.num_data_points // self.batch_sizes[a]):
                ind = np.random.choice(self.num_data_points, self.batch_sizes[a])
                batch_x, batch_y = self.x[ind], self.y[ind]
                model.train_model(batch_x, batch_y)
    
    def validate_models(self, run_number, epoch_number):
        #validate models
        for a, model in enumerate(self.models):
            mu, var = model.test_model(self.x, self.y)
            distance = torch.dist(torch.from_numpy(self.y).float(), mu)
            self.error_list[run_number, epoch_number, a] = distance

            # draw plot till now
            if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
                # mu, var = model.test_model(x, y)
                self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], var[:, 0], 'model '+ model.name, self.plot_colors[a])
        
        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
            plt.plot(self.x, self.y, 'black', label='ground truth')
            plt.title('models after '+ str(epoch_number) +' epochs in run number ' + str(run_number+1))
            plt.legend()
            plt.show()

    def drawPlotUncertainty(self, x, y, y_err, label, color):
        plt.plot(x, y, color, label=label)
        plt.fill_between(x,
                        y - y_err,
                        y + y_err,
                        facecolor=color, alpha=0.4, edgecolor='none')
