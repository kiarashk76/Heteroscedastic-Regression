import numpy as np
from ExperimentWithError import experimentWithError


class experiment_bias1(experimentWithError):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 10)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            self.noise[i] = 5 * np.random.normal(0.5*x[i], 0)
        y = 2 * x + self.noise
        mu = 2 * x
        self.x, self.y, self.mu = x, y, mu




class experiment_bias2(experimentWithError):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            if 2 < x[i] < 3:
                self.noise[i] = np.random.normal(2, 0)
        y = 2 * x + self.noise
        mu = 2 * x
        self.x, self.y, self.mu = x, y, mu


class experiment_bias3(experimentWithError):
    def __init__(self, params, experiment_name):
        experimentWithError.__init__(self, params, experiment_name)
        self.A = 0.5

    def create_dataset(self):
        # create the dataset
        range_data_points = (-4, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))

        y = self.A*(x ** 2)
        mu = self.A*(x ** 2)
        self.x, self.y, self.mu = x, y, mu


class experiment_bias4(experimentWithError):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            if 2 < x[i] < 3:
                self.noise[i] = x[i] + 1
        y = 2 * x + self.noise
        mu = 2 * x
        self.x, self.y, self.mu = x, y, mu


class experiment_bias5(experimentWithError):
    def __init__(self, params, experiment_name):
        experimentWithError.__init__(self, params, experiment_name)
        self.A = 0.5

    def create_dataset(self):
        # create the dataset
        range_data_points = (-2, 2)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], self.num_data_points*1), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, 1))
        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            self.noise[i] = np.random.normal(0, np.exp(- x[i]**2) )
        y = self.A*(x ** 2) + self.noise
        mu = self.A*(x ** 2)
        self.x, self.y, self.mu = x, y, mu