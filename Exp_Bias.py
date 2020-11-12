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
        x = np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        y = 5* (x ** 2 )
        self.x, self.y = x, y

class experiment_bias2(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 5)
        x = np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        y = 2*x + 0.5 
        self.x, self.y = x, y