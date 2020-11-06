import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Experiment import experiment

class experimentWithError(experiment):
    def __init__(self, params):
        experiment.__init__(self, params)
        self.hidden_layers_error = params['hidden_layers_error']

    def init_models(self):
        #initializing the models
        self.models = []
        for i in range(self.num_agents):
            params = {"hidden_layers_mu": self.hidden_layers_mu[i],
                      "hidden_layers_var": self.hidden_layers_var[i],
                      "hidden_layers_error": self.hidden_layers_error[i],
                      "batch_size": self.batch_sizes[i],
                      "step_size": self.step_sizes[i],
                      "name": self.names[i]}
            m = GeneralModelwithError(params)
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
        plt.title("error plot over all the runs")
        plt.legend()
        plt.show()          
    
    
    def validate_models(self, run_number, epoch_number):
        #validate models
        for a, model in enumerate(self.models):
            mu, var = model.test_model(self.x, self.y)
            distance = torch.dist(torch.from_numpy(self.y).float(), mu)
            self.error_list[run_number, epoch_number, a] = distance

            # draw plot till now
            if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
                # mu, var = model.test_model(x, y)
                error = model.test_error(self.x, self.y)
                self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], var[:, 0], 'model '+ model.name, self.plot_colors[a])
                self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], error[:, 0], 'model2 '+ model.name, self.plot_colors[a])
        
        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
            plt.plot(self.x, self.y, 'black', label='ground truth')
            plt.title('models after '+ str(epoch_number) +' epochs in run number ' + str(run_number+1))
            plt.legend()
            plt.show()