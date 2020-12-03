import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Experiment import experiment

class experiment_irreducible_error1(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            self.noise[i] = np.random.normal(0, 0.5*x[i])
        y = 2 * x + self.noise
        mu = 2 * x
        self.x, self.y, self.mu = x, y, mu



class experiment_irreducible_error2(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            self.noise[i] = np.random.normal(2, 0.5*x[i])
        y = 2 * x + self.noise
        mu = 2 * x
        self.x, self.y, self.mu = x, y, mu


class experiment_irreducible_error3(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            self.noise[i] = np.random.normal(1, np.abs(np.sin(2*x[i])))
        y = 2 * x + self.noise
        mu = 2 * x
        self.x, self.y, self.mu = x, y, mu


class experiment_irreducible_error4(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            if 2 < x[i] < 3:
                self.noise[i] = np.random.uniform(0, 3)
        y = 2 * x + self.noise
        mu = 2 * x
        self.x, self.y, self.mu = x, y, mu



class experiment_irreducible_error5(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        self.noise = np.zeros_like(x)
        y = np.zeros_like(x)
        for i in range(x.shape[0]):
            if 2 < x[i] < 4:
                self.noise[i] = np.random.uniform(-1, 1)
            if 0 <= x[i] <= 4:
                y[i] += np.sin(5*x[i]) + 2
            # if 0 <= x[i] <=2:
            #     y[i] += x[i]
        y += self.noise
        mu = np.sin(x)
        self.x, self.y, self.mu = x, y, mu

class experiment_irreducible_error6(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
                self.noise[i] = np.sin(x[i] * np.pi/2)
                if 0<x[i]<1:
                    self.noise[i] += np.random.uniform(-1,1)
                if 3<x[i]<4:
                    self.noise[i] += np.random.uniform(-1,1)

        y = np.ones_like(x) + self.noise
        mu = np.ones_like(x)
        self.x, self.y, self.mu = x, y, mu
