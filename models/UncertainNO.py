import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, Linear, MSELoss
from einops import rearrange, reduce
from laplace.lllaplace import DiagLLLaplace
from models.FNO1d import FNO1d
import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

class ModifiedDiagLLLaplace(DiagLLLaplace):
    def __init__(self, model, likelihood, *args, **kwargs):
        super().__init__(model, likelihood, *args, **kwargs)
    
    def _curv_closure(self, X, y, N):
        return super()._curv_closure(X, y.reshape(-1, 1), N)

class BayesianNO(nn.Module):
    def __init__(self, base_model_class, base_model_params):
        super().__init__()
        self.base_model_class = base_model_class
        self.base_model_params = base_model_params
        self.loss_func = None
        self.la = None

        # self.la = DiagLLLaplace(base_model, likelihood='regression')
        # self.la = ModifiedDiagLLLaplace(base_model, likelihood='regression')
    
    def fit(self, train_loader, valid_loader, **fit_params):
        base_model = self.base_model_class(last_layer_reshape=False, **self.base_model_params).to(device)
        base_model.fit(train_loader, valid_loader, **fit_params)
        base_model.last_layer_reshape=True

        self.la = ModifiedDiagLLLaplace(base_model, likelihood='regression')
        self.la.fit(train_loader)
        log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
        for i in range(1000):
            hyper_optimizer.zero_grad()
            neg_marglik = -self.la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
            neg_marglik.backward()
            hyper_optimizer.step()

        self.loss_func = base_model.loss_func
    
    def forward(self, x):
        b, p, t, _ = x.shape

        # Returns mean and variance
        mu, var = self.la(x)

        # Assumes diag
        mu = rearrange(mu, "(b p t) 1 -> b p t 1", b=b, p=p)
        var = rearrange(var, "(b p t) 1 1 -> b p t 1", b=b, p=p)

        return mu, var
    
    def parameters(self):
        return self.la.model.parameters()

    def test(self, test_loader, **test_params):
        batch_size = test_params.get("batch_size", 20)
        test_l2 = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                out, _ = self(x)
                test_l2 += self.loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        test_l2 /= len(test_loader.dataset)
        return {"loss": test_l2}


class EnsembleNO(nn.Module):
    def __init__(self, base_model_class, base_model_params, n_models=10):
        super().__init__()
        self.base_model_class = base_model_class
        self.base_model_params = base_model_params
        self.n_models = n_models
        # self.models_state_dict = []
        self.models = []
        self.loss_func = None

    def fit(self, train_loader, valid_loader, **fit_params):
        # train_loader should have shuffle=True
        for i in range(self.n_models):
            print("="*20 + f" Model {i} " + "=" * 20)
            base_model = self.base_model_class(**self.base_model_params).to(device)
            base_model.fit(train_loader, valid_loader, **fit_params)
            # self.models_state_dict.append(base_model.state_dict())
            base_model.eval()
            self.models.append(base_model)
        self.loss_func = base_model.loss_func

    def forward(self, x):
        # During inference only
        # if len(self.models_state_dict) == 0:
        if len(self.models) == 0:
            print("Models not trained. Use fit() function.")
            return

        out_list = []
        # for state_dict in self.models_state_dict:
        for base_model in self.models:
            # base_model = self.base_model_class(**self.base_model_params).to(device)
            # base_model.load_state_dict(state_dict)
            out = base_model(x)
            out_list.append(out)
        
        out_list = torch.stack(out_list) 

        return out_list.mean(dim=0), out_list.var(dim=0)
    
    def parameters(self):
        for base_model in self.models:
            # for p in state_dict.values():
            for p in base_model.parameters():
                yield p
                
    def finetune(self, finetune_loader, **fit_params):
        # After training only
        if len(self.models_state_dict) == 0:
            print("Models not trained. Use fit() function.")
            return

        finetuned_models_state_dict = []
        for i, state_dict in enumerate(self.models_state_dict):
            print("="*20 + f" Model {i} " + "=" * 20)
            base_model = self.base_model_class(**self.base_model_params).to(device)
            base_model.load_state_dict(state_dict)
            base_model.finetune(finetune_loader, **fit_params)
            finetuned_models_state_dict.append(base_model.state_dict())

        self.models_state_dict = finetuned_models_state_dict
    
    def test(self, test_loader, **test_params):
        batch_size = test_params.get("batch_size", 20)
        test_l2 = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                out, _ = self(x)
                test_l2 += self.loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        test_l2 /= len(test_loader.dataset)

        return {"loss": test_l2}


class MCDropoutNO(nn.Module):
    def __init__(self, base_model_class, base_model_params, dropout=0.1, n_dropouts=30):
        super().__init__()
        self.base_model_class = base_model_class
        self.base_model_params = base_model_params
        self.dropout = dropout
        self.n_dropouts = n_dropouts
        self.loss_func = None
        self.base_model = None
    
    def fit(self, train_loader, valid_loader, **fit_params):
        self.base_model = self.base_model_class(dropout=self.dropout, **self.base_model_params).to(device)
        self.base_model.fit(train_loader, valid_loader, **fit_params)
        self.loss_func = self.base_model.loss_func

    def forward(self, x):
        outs_list = []
        for i in range(self.n_dropouts):
            outs = self.base_model(x)
            outs_list.append(outs)
        outs_list = torch.stack(outs_list)
        return outs_list.mean(dim=0), outs_list.var(dim=0)
    
    def parameters(self):
        return self.base_model.parameters()

    def test(self, test_loader, **test_params):
        batch_size = test_params.get("batch_size", 20)
        test_l2 = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                out, _ = self(x)
                test_l2 += self.loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        test_l2 /= len(test_loader.dataset)

        return {"loss": test_l2}


class OutputVarNO(nn.Module):
    def __init__(self, base_model_class, base_model_params):
        super().__init__()
        self.base_model_class = base_model_class
        self.base_model_params = base_model_params
        self.loss_func = None
        self.base_model = None
    
    def fit(self, train_loader, valid_loader, **fit_params):
        self.base_model = self.base_model_class(output_var=True, **self.base_model_params).to(device)
        self.base_model.fit(train_loader, valid_loader, **fit_params)
        self.loss_func = self.base_model.loss_func
    
    def forward(self, x):
        return self.base_model(x)

    def parameters(self):
        return self.base_model.parameters()

    def test(self, test_loader, **test_params):
        batch_size = test_params.get("batch_size", 20)
        test_l2 = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                out, _ = self(x)
                # test_l2 += self.loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        test_l2 /= len(test_loader.dataset)
        return {"loss": test_l2}


if __name__ == '__main__':
    pass
    # # Usage
    # from models.FNO2d import FNO2d
    # x_train = torch.rand(1, 100, 20, 1)
    # y_train = torch.rand(1, 100, 20, 1)
    # train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
    #                                             batch_size=20, shuffle=True)

    # FNO2d_params = {"modes1": 12, "modes2": 12, "width": 32}
    # uq_model = BayesianNO(FNO2d, FNO2d_params)
    # uq_model = OutputVar(FNO2d, FNO2d_params)
    # uq_model = EnsembleNO(FNO2d, FNO2d_params, n_models=10)
    # uq_model = MCDropoutNO(FNO2d, FNO2d_params, dropout=0.1, n_dropouts=30)

    # uq_model.fit(train_loader, train_loader, epochs=10)
    # mu, var = uq_model(x_train.to(device))
    # print(mu.shape, var.shape)
    # results = uq_model.test(train_loader)
    # print(results)

