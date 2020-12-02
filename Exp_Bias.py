import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Experiment import experiment


class experiment_bias1(experiment):
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

    def train_models(self):
        for a, model in enumerate(self.models):
            for _ in range(self.num_data_points // self.batch_sizes[a]):
                ind = np.random.choice(self.num_data_points, self.batch_sizes[a])
                batch_x, batch_y, batch_mu = self.x[ind], self.y[ind], self.mu[ind]
                # give batch_mu so mu not being learned
                if self.mu_training[a]:
                    model.train_model(batch_x, batch_y, batch_mu=None)
                else:
                    model.train_model(batch_x, batch_y, batch_mu=batch_mu)

    def validate_models(self, run_number, epoch_number):
        # validate models
        ground_truth = False

        for a, model in enumerate(self.models):
            mu, var = model.test_model(self.x, self.y)
            if not self.mu_training[a]:
                mu = torch.from_numpy(self.mu).float()
            distance = torch.dist(torch.from_numpy(self.y).float(), mu)
            noise = (torch.from_numpy(self.y).float() - mu) ** 2
            sigma_distance = torch.dist(noise, var)
            self.error_list[run_number, epoch_number, a] = distance
            self.error_list_sigma[run_number, epoch_number, a] = sigma_distance
            self.learn_mu[run_number, epoch_number, a] = mu
            self.learn_var[run_number, epoch_number, a] = var
            # draw plot till now
            if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
                # mu, var = model.test_model(x, y)
                # error = model.test_error(self.x, self.y) #only for GeneralModelwithError
                fig, axs = plt.subplots(2, 1)
                if not ground_truth:
                    axs[0].plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
                    axs[0].title.set_text(
                        'models after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))
                    axs[1].plot(self.x, noise, 'ko', markersize=0.5, label='ground truth',
                                alpha=0.5)  # '''0.5* self.x'''
                    axs[1].title.set_text(
                        'sigma after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))

                self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], var[:, 0], 'model ' + model.name,
                                         self.plot_colors[a],
                                         axs[0])
                axs[1].plot(self.x[:, 0], var[:, 0])



        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
            plt.show()
            plt.close()

        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_save:
            plt.plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
            plt.title('models after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))
            plt.savefig('plots/' + self.experiment_name + f'{epoch_number:04}' + '.png')
            plt.close()


class experiment_bias2(experiment):
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

    def train_models(self):
        for a, model in enumerate(self.models):
            for _ in range(self.num_data_points // self.batch_sizes[a]):
                ind = np.random.choice(self.num_data_points, self.batch_sizes[a])
                batch_x, batch_y, batch_mu = self.x[ind], self.y[ind], self.mu[ind]
                # give batch_mu so mu not being learned
                if self.mu_training[a]:
                    model.train_model(batch_x, batch_y, batch_mu=None)
                else:
                    model.train_model(batch_x, batch_y, batch_mu=batch_mu)

    def validate_models(self, run_number, epoch_number):
        # validate models
        ground_truth = False

        for a, model in enumerate(self.models):
            mu, var = model.test_model(self.x, self.y)
            if not self.mu_training[a]:
                mu = torch.from_numpy(self.mu).float()
            distance = torch.dist(torch.from_numpy(self.y).float(), mu)
            noise = (torch.from_numpy(self.y).float() - mu) ** 2

            sigma_distance = torch.dist(noise, var)
            self.error_list[run_number, epoch_number, a] = distance
            self.error_list_sigma[run_number, epoch_number, a] = sigma_distance
            self.learn_mu[run_number, epoch_number, a] = mu
            self.learn_var[run_number, epoch_number, a] = var
            # draw plot till now
            if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
                # mu, var = model.test_model(x, y)
                # error = model.test_error(self.x, self.y) #only for GeneralModelwithError
                fig, axs = plt.subplots(2, 1)
                if not ground_truth:
                    axs[0].plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
                    axs[0].title.set_text(
                        'models after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))
                    axs[1].plot(self.x, noise, 'ko', markersize=0.5, label='ground truth',
                                alpha=0.5)  # '''0.5* self.x'''
                    axs[1].title.set_text(
                        'sigma after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))

                self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], var[:, 0], 'model ' + model.name,
                                         self.plot_colors[a],
                                         axs[0])
                axs[1].plot(self.x[:, 0], var[:, 0])



        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
            plt.show()
            plt.close()

        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_save:
            plt.plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
            plt.title('models after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))
            plt.savefig('plots/' + self.experiment_name + f'{epoch_number:04}' + '.png')
            plt.close()

class experiment_bias3(experiment):
    def __init__(self, params, experiment_name):
        experiment.__init__(self, params, experiment_name)
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
        mu = self.A*(x ** 2) + self.noise
        self.x, self.y, self.mu = x, y, mu

    def train_models(self):
        for a, model in enumerate(self.models):
            for _ in range(self.num_data_points // self.batch_sizes[a]):
                ind = np.random.choice(self.num_data_points, self.batch_sizes[a])
                batch_x, batch_y, batch_mu = self.x[ind], self.y[ind], self.mu[ind]
                # give batch_mu so mu not being learned
                if self.mu_training[a]:
                    model.train_model(batch_x, batch_y, batch_mu=None)
                else:
                    model.train_model(batch_x, batch_y, batch_mu=batch_mu)

    def validate_models(self, run_number, epoch_number):
        # validate models
        ground_truth = False

        for a, model in enumerate(self.models):
            mu, var = model.test_model(self.x, self.y)
            if not self.mu_training[a]:
                mu = torch.from_numpy(self.mu).float()
            distance = torch.dist(torch.from_numpy(self.y).float(), mu)
            noise = (torch.from_numpy(self.y).float() - mu) ** 2

            sigma_distance = torch.dist(noise, var)
            self.error_list[run_number, epoch_number, a] = distance
            self.error_list_sigma[run_number, epoch_number, a] = sigma_distance
            self.learn_mu[run_number, epoch_number, a] = mu
            self.learn_var[run_number, epoch_number, a] = var
            # draw plot till now
            if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
                # mu, var = model.test_model(x, y)
                # error = model.test_error(self.x, self.y) #only for GeneralModelwithError
                fig, axs = plt.subplots(2, 1)
                if not ground_truth:
                    axs[0].plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
                    axs[0].title.set_text(
                        'models after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))
                    axs[0].set_ylim(-10,10)

                    axs[1].plot(self.x, noise, 'ko', markersize=0.5, label='ground truth',
                                alpha=0.5)  # '''0.5* self.x'''
                    axs[1].title.set_text(
                        'sigma after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))

                self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], var[:, 0], 'model ' + model.name,
                                         self.plot_colors[a],
                                         axs[0])
                axs[1].plot(self.x[:, 0], var[:, 0])



        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
            plt.show()
            plt.close()

        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_save:
            plt.plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
            plt.title('models after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))
            plt.savefig('plots/' + self.experiment_name + f'{epoch_number:04}' + '.png')
            plt.close()


class experiment_bias4(experiment):
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

    def train_models(self):
        for a, model in enumerate(self.models):
            for _ in range(self.num_data_points // self.batch_sizes[a]):
                ind = np.random.choice(self.num_data_points, self.batch_sizes[a])
                batch_x, batch_y, batch_mu = self.x[ind], self.y[ind], self.mu[ind]
                # give batch_mu so mu not being learned
                if self.mu_training[a]:
                    model.train_model(batch_x, batch_y, batch_mu=None)
                else:
                    model.train_model(batch_x, batch_y, batch_mu=batch_mu)

    def validate_models(self, run_number, epoch_number):
        # validate models
        ground_truth = False

        for a, model in enumerate(self.models):
            mu, var = model.test_model(self.x, self.y)
            if not self.mu_training[a]:
                mu = torch.from_numpy(self.mu).float()
            distance = torch.dist(torch.from_numpy(self.y).float(), mu)
            noise = (torch.from_numpy(self.y).float() - mu) ** 2

            sigma_distance = torch.dist(noise, var)
            self.error_list[run_number, epoch_number, a] = distance
            self.error_list_sigma[run_number, epoch_number, a] = sigma_distance
            self.learn_mu[run_number, epoch_number, a] = mu
            self.learn_var[run_number, epoch_number, a] = var
            # draw plot till now
            if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
                # mu, var = model.test_model(x, y)
                # error = model.test_error(self.x, self.y) #only for GeneralModelwithError
                fig, axs = plt.subplots(2, 1)
                if not ground_truth:
                    axs[0].plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
                    axs[0].title.set_text(
                        'models after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))
                    axs[1].plot(self.x, noise, 'ko', markersize=0.5, label='ground truth',
                                alpha=0.5)  # '''0.5* self.x'''
                    axs[1].title.set_text(
                        'sigma after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))

                self.drawPlotUncertainty(self.x[:, 0], mu[:, 0], var[:, 0], 'model ' + model.name,
                                         self.plot_colors[a],
                                         axs[0])
                axs[1].plot(self.x[:, 0], var[:, 0])



        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_show:
            plt.show()
            plt.close()

        if epoch_number % self.plot_show_epoch_freq == 0 and self.plt_save:
            plt.plot(self.x, self.y, 'ko', markersize=0.5, label='ground truth', alpha=0.5)
            plt.title('models after ' + str(epoch_number) + ' epochs in run number ' + str(run_number + 1))
            plt.savefig('plots/' + self.experiment_name + f'{epoch_number:04}' + '.png')
            plt.close()