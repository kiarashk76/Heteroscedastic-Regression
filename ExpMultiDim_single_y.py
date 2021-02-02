import numpy as np
from Experiment import experiment

class MD_experiment_irreducible_linear_single_y(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], (self.num_data_points, self.parameters['data_dim'])), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, self.parameters['data_dim']))

        range_w = (0, 4)
        w = np.round(np.random.uniform(range_w[0], range_w[1], (self.parameters['data_dim'], 1)))
        mu = np.dot(x, w)

        self.noise = np.zeros_like(mu)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if 2 < x[i][j] < 3:
                    self.noise[i] = np.random.normal(0, 0.5 * x[i])
        
        y = mu + self.noise 

        self.x, self.y, self.mu = x, y, mu

class MD_experiment_rangeLBias_single_y(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], (self.num_data_points, self.parameters['data_dim'])), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, self.parameters['data_dim']))

        range_w = (0, 4)
        w = np.round(np.random.uniform(range_w[0], range_w[1], (self.parameters['data_dim'], 1)))
        mu = np.dot(x, w)

        self.noise = np.zeros_like(mu)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if 2 < x[i][j] < 3:
                    self.noise[i] = x[i] + 1
        
        y = mu + self.noise 

        self.x, self.y, self.mu = x, y, mu

class MD_experiment_irreducible_error_single_y(experiment):
    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], (self.num_data_points, self.parameters['data_dim'])), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, self.parameters['data_dim']))

        range_w = (0, 4)
        w = np.round(np.random.uniform(range_w[0], range_w[1], (self.parameters['data_dim'], 1)))
        mu = np.dot(x, w)

        self.noise = np.zeros_like(mu)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if 2 < x[i][j] < 3:
                    self.noise[i] = np.random.uniform(0, 3)
        
        y = mu + self.noise 

        self.x, self.y, self.mu = x, y, mu

class MD_experiment_quadraticBias_single_y(experiment):
    def __init__(self, params, experiment_name):
        experiment.__init__(self, params, experiment_name)
        self.A = 0.5

    def create_dataset(self):
        # create the dataset
        range_data_points = (0, 4)
        x = np.round(np.random.uniform(range_data_points[0], range_data_points[1], (self.num_data_points, self.parameters['data_dim'])), 3)
        x = np.reshape(np.sort(x), (self.num_data_points, self.parameters['data_dim']))

        range_w = (0, 4)
        w = np.round(np.random.uniform(range_w[0], range_w[1], (self.parameters['data_dim'], 1)))
        mu = np.dot(x**2, w)
        y = mu

        self.x, self.y, self.mu = x, y, mu