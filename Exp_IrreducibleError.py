import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Experiment import experiment

class experiment_irreducible_error(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 5)
        x = np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1)
        x = np.reshape(np.sort(x), (self.num_data_points,1))
        noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            noise[i] = np.random.normal(0, x[i]**2)
        y = 2 * x + noise
        mu = 2 * x
        self.x, self.y, self.mu = x, y, mu

    def train_models(self):
        for a, model in enumerate(self.models):
            for _ in range(self.num_data_points // self.batch_sizes[a]):
                ind = np.random.choice(self.num_data_points, self.batch_sizes[a])
                batch_x, batch_y, batch_mu = self.x[ind], self.y[ind], self.mu[ind]
                model.train_model(batch_x, batch_y)#, batch_mu=None)

    def validate_models(self, run_number, epoch_number):
        #validate models
        for a, model in enumerate(self.models):
            mu, var = model.test_model(self.x, self.y)
            distance = torch.dist(torch.from_numpy(self.y).float(), mu)
            self.error_list[run_number, epoch_number, a] = distance

            # draw plot till now
            if epoch_number % self.plot_show_epoch_freq == 0 and (self.plt_show or self.plt_save):
                # mu, var = model.test_model(x, y)
                # error = model.test_error(self.x, self.y)

                plt.plot(self.x, self.y, 'ko', label='ground truth', alpha=0.01)

                # drawPlotUncertainty(self.x[:, 0], mu[:, 0], var[:, 0], 'model '+ model.name, self.plot_colors[a])
                # mu = torch.from_numpy(self.mu)
                self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], var[:, 0], 'model '+ model.name, self.plot_colors[a])

                # print(var[:, 0].mean(), error[:, 0].mean())
                # drawPlotUncertainty(self.x[:, 0], mu[:, 0], error[:, 0], 'model2 '+ model.name, self.plot_colors[a])

        
        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
            plt.title('models after '+ str(epoch_number) +' epochs in run number ' + str(run_number+1))
            plt.legend()
            plt.show()
            plt.close()
        
        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_save:
            plt.title('models after '+ str(epoch_number) +' epochs in run number ' + str(run_number+1))
            plt.legend()
            plt.savefig('plots/irreducible_err/' + f'{epoch_number:04}' + '.png')
            plt.close()
