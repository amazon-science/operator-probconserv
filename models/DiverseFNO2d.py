import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import utils
from einops import rearrange
from models.FNO2d import SpectralConv2d
from models.FNO2d import FourierBlock
from itertools import combinations


device = "cuda" if torch.cuda.is_available() else "cpu" 

class DiverseFNO2d(nn.Module):
    def __init__(self,
                 modes1,
                 modes2,
                 width,
                 reg_loss,
                 n_layers=4,
                 n_outputs=32,
                 n_regularize=5,
                 bias_last=False,
                 lam=1.,
                 lb=0,
                 ub=1):
        super().__init__()


        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.lb = lb
        self.ub = ub
        self.n_outputs = n_outputs
        self.reg_loss = reg_loss
        self.lam = lam
        self.n_regularize = n_regularize

        self.padding = 9  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.fourier_layers = nn.Sequential(
            *[FourierBlock(self.modes1, self.modes2, self.width) for i in range(n_layers-1)],
            FourierBlock(self.modes1, self.modes2, self.width, activation=False)
            )

        self.fc1 = nn.Linear(self.width, 128)

        self.fc2 = nn.Linear(128, n_outputs, bias=bias_last)

        self.loss_func = utils.LpLoss(size_average=False)

    
    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x = self.fourier_layers(x)

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)

        intermediate_output = x

        x = self.fc2(x)

        return x.mean(dim=-1, keepdims=True), x.var(dim=-1, keepdims=True), x, intermediate_output

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(self.lb, self.ub, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(self.lb, self.ub, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def fit(self, train_loader, valid_loader, x_test=None, **fit_params):
        lr = fit_params.get("lr", 1e-3)
        step_size = fit_params.get("step_size", 50)
        gamma = fit_params.get("gamma", 0.5)
        epochs = fit_params.get("epochs", 200)
        warmup = fit_params.get("warmup", 10)
        
        n_regularize = self.n_regularize

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        best_valid_l2 = np.inf

        for epoch in range(epochs):
            self.train()
            train_l2 = 0
            train_reg = 0
            for batch in train_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                _, _, out, z = self(x)
                
                l2 = 0.
                for i in range(out.shape[-1]):
                    l2 = l2 + self.loss_func(out[:, :, :, i], y)
                l2 = l2 / out.shape[-1]

                # l2 = self.loss_func(out.mean(dim=-1), y)
                
                if epoch >= warmup:
                    reg = self.regularization(n_regularize=n_regularize, reg_loss=self.reg_loss, x_test=x_test, z=z)
                else:
                    reg = torch.tensor(0.)
                
                reg = self.lam * reg
                loss = l2 - reg

                loss.backward() 
                optimizer.step()
                
                train_l2 += l2.item()
                train_reg += reg.item()
                
            train_l2 /= len(train_loader.dataset)
            train_reg /= len(train_loader.dataset)
            
            scheduler.step()
            if valid_loader is not None:
                valid_l2 = self.test(valid_loader, **fit_params)["loss"]
            else:
                valid_l2 = train_l2

            saved = "" 
            if valid_l2 < best_valid_l2:
                best_valid_l2 = valid_l2
                best_model_state_dict = deepcopy(self.state_dict())
                saved = "(saved)"

            # wandb.log({"train_loss": train_l2, "valid_loss": valid_l2, "reg": train_reg})
            print(f"Epoch {epoch}: Train loss={train_l2:.6f} Reg={train_reg:.4f}, Validation loss={valid_l2:.6f} {saved}")
            
        self.load_state_dict(best_model_state_dict)
        train_l2 = self.test(train_loader, **fit_params)["loss"]
        if valid_loader is not None:
            valid_l2 = self.test(valid_loader, **fit_params)["loss"]
        else:
            valid_l2 = train_l2
        print(f"Finished training with best train loss: {train_l2:.6f} and validation loss: {valid_l2:.6f}")

    def test(self, test_loader, **test_params):
        self.eval()
        test_l2 = 0.0
        test_mse_by_t = None
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                _, _, out, _ = self(x)
                
                l2 = 0.
                for i in range(out.shape[-1]):
                    l2 = l2 + self.loss_func(out[:, :, :, i], y).item()
                l2 = l2 / out.shape[-1]
                
                test_l2 += l2
                

        test_l2 /= len(test_loader.dataset)

        return {"loss": test_l2}
    
    def regularization(self, n_regularize, reg_loss, x_test, z=None):

        ms = np.random.choice(self.n_outputs, n_regularize)
        if reg_loss == "weights_l2":
            reg = 0.
            for m1, m2 in combinations(ms, 2):
                reg_ = ((self.fc2.weight[m1] - self.fc2.weight[m2])**2).sum()
                reg = reg + reg_
        elif reg_loss == "gradients_l2":
            reg = 0.
            ZtZ = torch.einsum("b x t c, b x t d -> b c d", z, z)
            for m1, m2 in combinations(ms, 2):
                pre_norm = (self.fc2.weight[m1] - self.fc2.weight[m2]).T @ ZtZ
                reg_ = pre_norm.norm(2, dim=1).mean()
                reg = reg + reg_
        elif reg_loss == "std_gradients_l2":
            z_ = rearrange(z, "b nx nt c -> b (nx nt) c")
            z_mean = z_.mean(dim=1, keepdims=True)
            z_std = z_.std(dim=1, keepdims=True)
            z_normalized = (z_ - z_mean) / z_std
            ZtZ = torch.einsum("b x c, b x d -> b c d", z_normalized, z_normalized) / (z_.shape[1] - 1)

            reg = 0.
            for m1, m2 in combinations(ms, 2):
                pre_norm = (self.fc2.weight[m1] - self.fc2.weight[m2]).T @ ZtZ
                reg_ = pre_norm.norm(2, dim=1).mean()
                reg = reg + reg_
        elif reg_loss.startswith("outputs_l2"):
            _, _, out_test, _ = self(x_test)
            reg = 0.
            for m1, m2 in combinations(ms, 2):
                out1 = out_test[:, :, :, m1]
                out2 = out_test[:, :, :, m2] 

                norm_out1 = out1 / out1.norm(dim=[1,2], keepdim=True)
                norm_out2 = out2 / out2.norm(dim=[1,2], keepdim=True)
                reg_ = self.loss_func.abs(norm_out1, norm_out2)

                if reg_loss.endswith("deriv"):
                    out1_grad = torch.gradient(out1, dim=1)[0]
                    out2_grad = torch.gradient(out2, dim=1)[0]

                    norm_out1_grad = out1_grad / out1_grad.norm(dim=[1,2], keepdim=True)
                    norm_out2_grad = out2_grad / out2_grad.norm(dim=[1,2], keepdim=True)
                    
                    reg_ = reg_ + self.loss_func.abs(norm_out1_grad, norm_out2_grad)

                reg = reg + reg_

        elif reg_loss == "std_outputs_l2":
            _, _, out_test, _ = self(x_test)
            reg = 0.
            for m1, m2 in combinations(ms, 2):
                mean1 = out_test[:, :, :, m1].mean(dim=0, keepdims=True)
                mean2 = out_test[:, :, :, m2].mean(dim=0, keepdims=True)
                std1 = out_test[:, :, :, m1].std(dim=0, keepdims=True)
                std2 = out_test[:, :, :, m2].std(dim=0, keepdims=True)

                norm_out1 = (out_test[:, :, :, m1] - mean1) / std1
                norm_out2 = (out_test[:, :, :, m2] - mean2) / std2

                reg_ = self.loss_func.abs(norm_out1, norm_out2)
                reg = reg + reg_
        else:
            raise NotImplementedError
        
        l = len(ms)
        return (2 * reg) / (l * (l - 1))


