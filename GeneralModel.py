# TWO GENERALMODEL - FIX
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from Models import ErrorNetwork, StateTransitionModel
from Models import StateTransitionModelSeparate, ModelError

class GeneralModel():
    def __init__(self, params):
        self.hidden_layers_mu = params['hidden_layers_mu']
        self.hidden_layers_var = params['hidden_layers_var']
        self.data_dim = params['data_dim']
        self.batch_size = params['batch_size']
        self.step_size = params['step_size']
        self.name = params['name']

        self.__create_networks()
    
    def __create_networks(self):
        self.model = StateTransitionModelSeparate(self.hidden_layers_mu, self.hidden_layers_var)#, self.data_dim)

    def __model_output(self, batch_x):
        with torch.no_grad():
            mu, sigma = self.model(batch_x.float())
        return mu, sigma

    def train_model(self, batch_x, batch_y, loss_type='1', batch_mu=None):#loss_type = '1', '2', '3'
        x = torch.from_numpy(batch_x).float()
        y = torch.from_numpy(batch_y).float()

        pred, var = self.model(x)
        assert pred.shape == x.shape, str(pred.shape) + str(var.shape) + str(x.shape)

        if batch_mu is None: # mu is being trained as well
            if loss_type == '1' :
                loss = torch.mean(((pred - y) ** 2) / (2 * var) + 0.5 * torch.log(var))
            elif loss_type == '2':
                A = (pred-y).unsqueeze(2)
                loss = torch.mean(torch.matmul(torch.matmul(A.permute(0,2,1), var), A).squeeze(2).squeeze(1) + torch.logdet(var))
            elif loss_type == '3':
                loss = torch.mean((pred - y) ** 2)
        else:
            mu = torch.from_numpy(batch_mu).float()
            if loss_type == '1':
                loss = torch.mean(((mu - y) ** 2) / (2 * var) + 0.5 * torch.log(var))
            else:
                raise NotImplementedError("other loss functions hasn't been implemented yet!")
        loss.backward()
        if torch.isnan(loss):
            print("loss is nan")

        optimizer = optim.Adam(self.model.parameters(), lr=self.step_size)
        # optimizer = optim.SGD(model.parameters(), lr=step_size)

        optimizer.step()
        optimizer.zero_grad()

        
    def test_model(self, batch_x, batch_y):
        x = torch.from_numpy(batch_x).float()
        y = torch.from_numpy(batch_y).float()
        mu, sigma = self.__model_output(x)
        # trace = torch.diagonal(sigma, dim1=1, dim2=2).sum(-1)
        return mu, sigma


# general model with error network

class GeneralModel():
    def __init__(self, params):
        self.hidden_layers_mu = params['hidden_layers_mu']
        self.hidden_layers_var = params['hidden_layers_var']
        self.data_dim = params['data_dim']
        self.batch_size = params['batch_size']
        self.step_size = params['step_size']
        self.name = params['name']

        self.__create_networks()
    
    def __create_networks(self):
        self.model = StateTransitionModelSeparate(self.hidden_layers_mu, self.hidden_layers_var)#, self.data_dim)
        params = {'model':self.model,
                  'batch_size':self.batch_size,
                  'step_size': self.step_size,
                  'name':'error_network',
                  'hidden_layers':self.hidden_layers_var}
        self.error_network = ErrorNetwork(params)  
        self.error_network.__create_networks()

    def __model_output(self, batch_x):
        with torch.no_grad():
            mu, sigma = self.model(batch_x.float())
        return mu, sigma

    def train_model(self, batch_x, batch_y, loss_type='1'):#loss_type = '1', '2', '3'
        x = torch.from_numpy(batch_x).float()
        y = torch.from_numpy(batch_y).float()

        pred, var = self.model(x)
        assert pred.shape == x.shape, str(pred.shape) + str(var.shape) + str(x.shape)
      
        if loss_type == '1' :
            loss = torch.mean(((pred - y) ** 2) / (2 * var) + 0.5 * torch.log(var))
        elif loss_type == '2':
            A = (pred-y).unsqueeze(2)
            loss = torch.mean(torch.matmul(torch.matmul(A.permute(0,2,1), var), A).squeeze(2).squeeze(1) + torch.logdet(var))
        elif loss_type == '3':
            loss = torch.mean((pred - y) ** 2)
        
        loss.backward()
        if torch.isnan(loss):
            print("loss is nan")

        optimizer = optim.Adam(self.model.parameters(), lr=self.step_size)
        # optimizer = optim.SGD(model.parameters(), lr=step_size)

        optimizer.step()
        optimizer.zero_grad()

        self.error_network.train_model(batch_x, batch_y)
        
    def test_model(self, batch_x, batch_y):
        x = torch.from_numpy(batch_x).float()
        y = torch.from_numpy(batch_y).float()
        mu, sigma = self.__model_output(x)
        # trace = torch.diagonal(sigma, dim1=1, dim2=2).sum(-1)
        return mu, sigma