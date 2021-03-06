# TWO GENERALMODEL - FIX
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Models import ErrorNetwork, StateTransitionModel, StateTransitionModelSeparateD_Dim
from Models import StateTransitionModelSeparate, ModelError

class GeneralModel():
    def __init__(self, params):
        self.hidden_layers_mu = params['hidden_layers_mu']
        self.hidden_layers_var = params['hidden_layers_var']
        self.data_dim = params['data_dim']
        self.batch_size = params['batch_size']
        self.step_size = params['step_size']
        self.name = params['name']
        self.bias_available = params['bias_available']
        self.loss_type = params['loss_type']

        self.__create_networks()

    def __del__(self):
        del self.model
    
    def __create_networks(self):

        # self.model = StateTransitionModelSeparateD_Dim(self.hidden_layers_mu, self.hidden_layers_var,
        #                                           bias_available=self.bias_available, self.data_dim)
        #self.model = StateTransitionModelSeparate(self.hidden_layers_mu, self.hidden_layers_var)
        self.model = StateTransitionModelSeparateD_Dim(self.hidden_layers_mu, self.hidden_layers_var,
                                                         self.data_dim)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.step_size)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.step_size)

    def __model_output(self, batch_x):
        with torch.no_grad():
            mu, sigma = self.model(batch_x.float())
        return mu, sigma

    def train_model(self, batch_x, batch_y, batch_mu=None, batch_var=None, episode_num=None):#loss_type = '1', '2', '3'
        loss_type = self.loss_type
        x = torch.from_numpy(batch_x).float().to(self.device)
        y = torch.from_numpy(batch_y).float().to(self.device)

        pred, var = self.model(x, episode_num)
        assert pred.shape == x.shape, str(pred.shape) + str(var.shape) + str(x.shape)
        if batch_mu is None: # mu is being trained as well
            if loss_type == '1' :
                # loss = torch.mean(((pred - y) ** 2) / (2 * batch_var) + 0.5 * torch.log(batch_var))
                loss = torch.mean(((pred - y) ** 2) / (2 * var) + 0.5 * torch.log(var))
            elif loss_type == '2':
                A = (pred-y).unsqueeze(2)
                inv_var = torch.inverse(var)
                loss = torch.mean(torch.matmul(torch.matmul(A.permute(0,2,1), inv_var), A).squeeze(2).squeeze(1) + torch.logdet(var))
            elif loss_type == '3':
                loss = torch.mean((pred - y) ** 2)
        else:
            mu = torch.from_numpy(batch_mu).float()
            if loss_type == '1':
                loss = torch.mean(((mu - y) ** 2) / (2 * var) + 0.5 * torch.log(var))
            else:
                raise NotImplementedError("other loss functions hasn't been implemented yet!")
        self.optimizer.zero_grad()
        loss.backward()
        if torch.isnan(loss):
            print(torch.log(var), (pred - y) ** 2, (2 * var))
            print("loss is nan")
            exit(0)

        self.optimizer.step()


        
    def test_model(self, batch_x, batch_y):
        x = torch.from_numpy(batch_x).float().to(self.device)
        y = torch.from_numpy(batch_y).float().to(self.device)
        mu, sigma = self.__model_output(x)
        # trace = torch.diagonal(sigma, dim1=1, dim2=2).sum(-1)
        return mu, sigma


# general model with error network

class GeneralModelwithError():
    def __init__(self, params):
        self.hidden_layers_mu = params['hidden_layers_mu']
        self.data_dim = params['data_dim']
        self.batch_size = params['batch_size']
        self.step_size = params['step_size']
        self.name = params['name']
        self.bias_available = params['bias_available']
        self.hidden_layers_error = params['hidden_layers_error']
        self.loss_type = params['loss_type']

        self.__create_networks()
    
    def __create_networks(self):
        self.model = StateTransitionModelSeparate(self.hidden_layers_mu, [],
                                                  bias_available=self.bias_available)#, self.data_dim)
        params = {'model':self.model,
                  'batch_size':self.batch_size,
                  'step_size': self.step_size,
                  'name':'error_network',
                  'hidden_layers':self.hidden_layers_error}
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.step_size)

        self.error_network = ErrorNetwork(params)

    def __model_output(self, batch_x):
        with torch.no_grad():
            mu, sigma = self.model(batch_x.float())
        return mu, sigma

    def train_model(self, batch_x, batch_y, batch_mu=None, batch_var=None):#loss_type = '1', '2', '3'
        x = torch.from_numpy(batch_x).float()
        y = torch.from_numpy(batch_y).float()
        loss_type = self.loss_type

        pred, var = self.model(x)
        assert pred.shape == x.shape, str(pred.shape) + str(var.shape) + str(x.shape)

        # for i in var:
        #     if torch.isnan(i):
        #         print("khar")
        #         pred, var = self.model(x)
        if batch_mu is None:  # mu is being trained as well
            if loss_type == '1':
                # loss = torch.mean(((pred - y) ** 2) / (2 * batch_var) + 0.5 * torch.log(batch_var))
                loss = torch.mean(((pred - y) ** 2) / (2 * var) + 0.5 * torch.log(var))
            elif loss_type == '2':
                A = (pred - y).unsqueeze(2)
                loss = torch.mean(
                    torch.matmul(torch.matmul(A.permute(0, 2, 1), var), A).squeeze(2).squeeze(1) + torch.logdet(var))
            elif loss_type == '3':
                loss = torch.mean((pred - y) ** 2)
            self.optimizer.zero_grad()
            loss.backward()
            if torch.isnan(loss):
                print(torch.log(var), (pred - y) ** 2, (2 * var))
                print("loss is nan")
                exit(0)
            self.optimizer.step()
            batch_y_hat = pred.detach()
        else:
            batch_y_hat = batch_mu
        self.error_network.train_error(batch_x, batch_y_hat, batch_y)



    def test_model(self, batch_x, batch_y):
        x = torch.from_numpy(batch_x).float()
        y = torch.from_numpy(batch_y).float()
        mu, sigma = self.__model_output(x)
        # trace = torch.diagonal(sigma, dim1=1, dim2=2).sum(-1)
        return mu, sigma