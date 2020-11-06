import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Experiment import experiment

class experiment_bias(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (-2, 2)
        x = np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points)
        x = np.reshape(np.sort(x), (self.num_data_points))
        y = x ** 2 
        self.x, self.y = x, y
    
    def validate_models(self, run_number, epoch_number):
        #validate models
        for a, model in enumerate(self.models):
            mu, var = model.test_model(self.x, self.y)
            distance = torch.dist(torch.from_numpy(self.y).float(), mu)
            self.error_list[run_number, epoch_number, a] = distance

            # draw plot till now
            if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
                # mu, var = model.test_model(x, y)
                plt.plot(self.x, self.y, 'k', label='ground truth')

                self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], var[:, 0], 'model '+ model.name, self.plot_colors[a])
        
        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
            plt.title('models after '+ str(epoch_number) +' epochs in run number ' + str(run_number+1))
            plt.legend()
            plt.show()


class experiment_bias2(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 5)
        x = np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        y = 2*x + 0.5 
        self.x, self.y = x, y
    
    def validate_models(self, run_number, epoch_number):
        #validate models
        for a, model in enumerate(self.models):
            mu, var = model.test_model(self.x, self.y)
            distance = torch.dist(torch.from_numpy(self.y).float(), mu)
            self.error_list[run_number, epoch_number, a] = distance

            # draw plot till now
            if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
                # mu, var = model.test_model(x, y)
                plt.plot(self.x, self.y, 'k', label='ground truth')
                self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], var[:, 0], 'model '+ model.name, self.plot_colors[a])
        
        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
            plt.title('models after '+ str(epoch_number) +' epochs in run number ' + str(run_number+1))
            plt.legend()
            plt.show()