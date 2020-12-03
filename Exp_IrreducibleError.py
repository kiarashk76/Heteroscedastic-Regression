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
                    axs[0].set_ylim(-2,6)
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