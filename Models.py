# State Transition Model (with Learned Variance)
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy

# 1-dim
# separated from the last layer
class StateTransitionModel(nn.Module):
    def __init__(self,num_hidden):
        super(StateTransitionModel, self).__init__()
        self.layers_list = []
        self.vlayers_list = []
        if len(num_hidden) == 0:
            self.mu = nn.Linear(1, 1)
            self.var = nn.Linear(1, 1)
            torch.nn.init.xavier_uniform_(self.mu.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.var.weight, gain=1.0)
        else:
            for i, num in enumerate(num_hidden):
                if i == 0:
                    l = nn.Linear(1, num)
                    torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
                    self.layers_list.append(l)
                    self.add_module('hidden_layer_' + str(i), l)

                    vl = nn.Linear(1, num)
                    torch.nn.init.xavier_uniform_(vl.weight, gain=1.0)
                    self.vlayers_list.append(vl)
                    self.add_module('vhidden_layer_' + str(i), vl)
                else:
                    l = nn.Linear(num_hidden[i-1], num)
                    torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
                    self.layers_list.append(l)
                    self.add_module('hidden_layer_' + str(i), l)

                    vl = nn.Linear(num_hidden[i - 1], num)
                    torch.nn.init.xavier_uniform_(vl.weight, gain=1.0)
                    self.vlayers_list.append(vl)
                    self.add_module('vhidden_layer_' + str(i), vl)
            self.mu = nn.Linear(num_hidden[-1], 1)
            self.var = nn.Linear(num_hidden[-1], 1)
            torch.nn.init.xavier_uniform_(self.mu.weight, gain=1.0)
            torch.nn.init.xavier_uniform_(self.var.weight, gain=1.0)


    def forward(self, x):
        l = x
        vl = x
        for i, lay in enumerate(zip(self.layers_list, self.vlayers_list)):
            layer = lay[0]
            vlayer = lay[1]
            l = torch.relu(layer(l))
            vl = torch.relu(vlayer(vl))
        mu = self.mu(l)
        var = torch.log(1 + torch.exp(self.var(vl)))
        return mu, var

# separated from the begining
class StateTransitionModelSeparate(nn.Module):
    def __init__(self, num_hidden_mu, num_hidden_var, bias_available=True):
        super(StateTransitionModelSeparate, self).__init__()
        self.layers_list = []
        self.vlayers_list = []
        if len(num_hidden_mu) == 0:
            self.mu = nn.Linear(1, 1, bias=bias_available)
            torch.nn.init.xavier_uniform_(self.mu.weight, gain=1.0)
        else:
            for i, num in enumerate(num_hidden_mu):
                if i == 0:
                    l = nn.Linear(1, num)
                else:
                    l = nn.Linear(num_hidden_mu[i-1], num)
                torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
                self.layers_list.append(l)
                self.add_module('hidden_layer_' + str(i), l)

            self.mu = nn.Linear(num_hidden_mu[-1], 1)
            torch.nn.init.xavier_uniform_(self.mu.weight, gain=1.0)

        if len(num_hidden_var) == 0:
            self.var = nn.Linear(1, 1)
            torch.nn.init.xavier_uniform_(self.var.weight, gain=1.0)
        else:
            for i, num in enumerate(num_hidden_var):
                if i == 0:
                    vl = nn.Linear(1, num)
                else:
                    vl = nn.Linear(num_hidden_var[i - 1], num)
                torch.nn.init.xavier_uniform_(vl.weight, gain=1.0)
                self.vlayers_list.append(vl)
                self.add_module('vhidden_layer_' + str(i), vl)

            self.var = nn.Linear(num_hidden_var[-1], 1)
            torch.nn.init.xavier_uniform_(self.var.weight, gain=1.0)


    def forward(self, x):
        l = copy.copy(x)
        vl = copy.copy(x)
        for i, lay in enumerate(self.layers_list):
            layer = lay
            l = torch.tanh(layer(l))
        for i, lay in enumerate(self.vlayers_list):
            vlayer = lay
            vl = torch.tanh(vlayer(vl))
        mu = self.mu(l)
        var = F.softplus(self.var(vl)) + 10**-6
        # var = torch.relu(self.var(vl)) + 0.1
        return mu, var


# d-dim
# separated from the begining
class StateTransitionModelSeparateD_Dim(nn.Module):
    def __init__(self, num_hidden_mu, num_hidden_var, d):
        super(StateTransitionModelSeparate, self).__init__()
        self.layers_list = []
        self.vlayers_list = []
        if len(num_hidden_mu) == 0:
            self.mu = nn.Linear(d, d, bias=False)
            torch.nn.init.xavier_uniform_(self.mu.weight, gain=1.0)
        else:
            for i, num in enumerate(num_hidden_mu):
                if i == 0:
                    l = nn.Linear(d, num)
                    torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
                    self.layers_list.append(l)
                    self.add_module('hidden_layer_' + str(i), l)
                else:
                    l = nn.Linear(num_hidden_mu[i-1], num)
                    torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
                    self.layers_list.append(l)
                    self.add_module('hidden_layer_' + str(i), l)
            self.mu = nn.Linear(num_hidden_mu[-1], d)
            torch.nn.init.xavier_uniform_(self.mu.weight, gain=1.0)

        if len(num_hidden_var) == 0:
            self.var = nn.Linear(d, d)
            torch.nn.init.xavier_uniform_(self.var.weight, gain=1.0)
        else:
            for i, num in enumerate(num_hidden_var):
                if i == 0:
                    vl = nn.Linear(d, num)
                    torch.nn.init.xavier_uniform_(vl.weight, gain=1.0)
                    self.vlayers_list.append(vl)
                    self.add_module('vhidden_layer_' + str(i), vl)
                else:
                    vl = nn.Linear(num_hidden_var[i - 1], num)
                    torch.nn.init.xavier_uniform_(vl.weight, gain=1.0)
                    self.vlayers_list.append(vl)
                    self.add_module('vhidden_layer_' + str(i), vl)
            self.var = nn.Linear(num_hidden_var[-1], d)
            torch.nn.init.xavier_uniform_(self.var.weight, gain=1.0)


    def forward(self, x):
        l = x
        vl = x
        for i, lay in enumerate(self.layers_list):
            layer = lay
            l = torch.relu(layer(l))
        for i, lay in enumerate(self.vlayers_list):
            vlayer = lay
            vl = torch.relu(vlayer(vl))
        mu = self.mu(l)
        var = torch.log(1 + torch.exp(self.var(vl)))
        var = torch.diag_embed(var)
        return mu, var


# MSE Network
class ModelError(nn.Module):
    def __init__(self, num_hidden):
        super(ModelError, self).__init__()
        self.layers_list = []
        if len(num_hidden) == 0:
            self.head = nn.Linear(1, 1, bias=False)
            torch.nn.init.xavier_uniform_(self.head.weight, gain=1.0)
        else:
            for i, num in enumerate(num_hidden):
                if i == 0:
                    l = nn.Linear(1, num)
                    torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
                    self.layers_list.append(l)
                    self.add_module('hidden_layer_' + str(i), l)
                else:
                    l = nn.Linear(num_hidden[i-1], num)
                    torch.nn.init.xavier_uniform_(l.weight, gain=1.0)
                    self.layers_list.append(l)
                    self.add_module('hidden_layer_' + str(i), l)
            self.head = nn.Linear(num_hidden[-1], 1)
            torch.nn.init.xavier_uniform_(self.mu.weight, gain=1.0)


    def forward(self, x):
        l = x
        for i, lay in enumerate(self.layers_list):
            layer = lay
            l = torch.relu(layer(l))
        return self.head(l)
        # return torch.log(1 + torch.exp(self.head(l)))


# Error Network
class ErrorNetwork():
    def __init__(self, params):
        self.model = params['model']
        self.batch_size = params['batch_size']
        self.step_size = params['step_size']
        self.name = params['name']
        self.hidden_layers = params['hidden_layers']

        self.__create_networks()
    
    def __create_networks(self):
        self.error = ModelError(self.hidden_layers)

    def __error_output(self, batch_x):
        with torch.no_grad():
            error = self.error(batch_x.float())
        return error
    
    def __model_output(self, batch_x):
        with torch.no_grad():
            mu, sigma = self.model(batch_x.float())
        return mu, sigma

    def train_error(self, batch_x, batch_y):
        x = torch.from_numpy(batch_x).float()
        y = torch.from_numpy(batch_y).float()
        y_hat, _ = self.__model_output(x)
        error = self.error(x)
        assert error.shape == x.shape, str(error.shape) + str(x.shape)
        target = (y - y_hat) ** 2
        loss = torch.mean((error - target) ** 2)
        loss.backward()
        if torch.isnan(loss):
            print("loss is nan")

        optimizer = optim.Adam(self.error.parameters(), lr=self.step_size)
        # optimizer = optim.SGD(model.parameters(), lr=step_size)

        optimizer.step()
        optimizer.zero_grad()
  
    def test_error(self, batch_x, batch_y):
        x = torch.from_numpy(batch_x).float()
        y = torch.from_numpy(batch_y).float()
        error = self.__error_output(x)

        return error