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
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
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