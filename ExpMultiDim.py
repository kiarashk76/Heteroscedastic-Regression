import numpy as np
from ExperimentWithError import experimentWithError

class MD_experiment_irreducible_linear(experimentWithError):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], (self.num_data_points, self.num_data_points)), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, self.num_data_points, 1))

        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                self.noise[i][j] = np.random.normal(0, 0.5*x[i][j])

        y = 2 * x + self.noise
        mu = 2 * x
        self.x, self.y, self.mu = x, y, mu

class MD_experiment_rangeLBias(experimentWithError):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], (self.num_data_points, self.num_data_points)), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, self.num_data_points, 1))

        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if 2 < x[i][j] < 3:
                    self.noise[i][j] = x[i][j] + 1
        y = 2 * x + self.noise
        mu = 2 * x
        self.x, self.y, self.mu = x, y, mu

class MD_experiment_irreducible_error(experimentWithError):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], (self.num_data_points, self.num_data_points)), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, self.num_data_points, 1))

        self.noise = np.zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if 2 < x[i][j] < 3:
                    self.noise[i][j] = np.random.uniform(0, 3)
        y = 2 * x + self.noise
        mu = 2 * x +
        self.x, self.y, self.mu = x, y, mu

class MD_experiment_quadraticBias(experimentWithError):
    def __init__(self, params, experiment_name):
        experimentWithError.__init__(self, params, experiment_name)
        self.A = 0.5

    def create_dataset(self):
        # create the dataset
        range_data_points = (-4, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], (self.num_data_points, self.num_data_points)), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, self.num_data_points, 1))

        y = self.A * (x ** 2)
        mu = self.A * (x ** 2)
        self.x, self.y, self.mu = x, y, mu        