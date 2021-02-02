import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Experiment import experiment


class experiment_homo1(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (-2, 2)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            self.noise[i] = np.random.normal(0, 0.5)
        y = x ** 3 + self.noise
        mu = x ** 3
        self.x, self.y, self.mu = x, y, mu