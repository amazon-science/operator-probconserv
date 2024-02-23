import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import utils
from einops import rearrange

# This code is borrowed from FNO git repository: https://github.com/zongyi-li/fourier_neural_operator


device = "cuda" if torch.cuda.is_available() else "cpu" 

################################################################
# 2D fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, drop_modes=0., drop_channels=0.):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        
        self.binomial_modes = torch.distributions.Binomial(self.modes1, probs=torch.tensor(1-drop_modes))

        self.n_drop_channels = int(drop_channels * self.in_channels)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        n_active_modes = self.binomial_modes.sample().int()
        channel_mask = torch.ones_like(self.weights1)
        perm = torch.randperm(self.in_channels)
        idx = perm[:self.n_drop_channels]
        channel_mask[idx, :, :] = 0.
        new_weights1 = self.weights1 * channel_mask
        new_weights2 = self.weights2 * channel_mask
        # n_active_modes = self.modes1
        # new_weights1 = self.weights1
        # new_weights2 = self.weights2

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :n_active_modes, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :n_active_modes, :self.modes2], new_weights1[:, :, :n_active_modes])
        out_ft[:, :, -n_active_modes:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -n_active_modes:, :self.modes2], new_weights2[:, :, :n_active_modes])

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FourierBlock(nn.Module):
    def __init__(self, modes1, modes2, width, drop_modes=0., drop_channels=0., activation=True):
        super().__init__()

        self.conv = SpectralConv2d(width, width, modes1, modes2, drop_modes, drop_channels)
        self.w = nn.Conv2d(width, width, 1)
        self.activation = activation

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        if self.activation:
            x = F.gelu(x)
        return x


class FNO2d(nn.Module):
    def __init__(self,
                 modes1,
                 modes2,
                 width,
                 n_layers=4,
                 dropout=0,
                 drop_modes=0.,
                 drop_channels=0.,
                 output_var=False,
                 n_outputs=1,
                 last_layer_reshape=False,
                 lb=0,
                 ub=1):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.lb = lb
        self.ub = ub
        self.dropout = dropout
        self.output_var = output_var
        self.n_outputs = n_outputs

        self.padding = 9  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.fourier_layers = nn.Sequential(
            *[FourierBlock(self.modes1, self.modes2, self.width, drop_modes, drop_channels) for i in range(n_layers-1)],
            FourierBlock(self.modes1, self.modes2, self.width, drop_modes, drop_channels, activation=False)
            )

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 2*n_outputs if self.output_var else n_outputs)

        if self.output_var:
            self.loss_func = utils.nll_mu_var
        else:
            self.loss_func = utils.LpLoss(size_average=False)

        self.last_layer_reshape = last_layer_reshape
    
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
        x = F.dropout(x, training=True, p=self.dropout) 
        x = F.gelu(x)


        # Last layer reshape for Laplace approximation
        if self.last_layer_reshape:
            x = rearrange(x, 'b p t d -> (b p t) d')

        x = self.fc2(x)

        if self.output_var:
            mu, var = torch.split(x, self.n_outputs, dim=-1)
            return mu, F.softplus(var)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(self.lb, self.ub, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(self.lb, self.ub, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def fit(self, train_loader, valid_loader, **fit_params):
        lr = fit_params.get("lr", 1e-3)
        step_size = fit_params.get("step_size", 50)
        gamma = fit_params.get("gamma", 0.5)
        epochs = fit_params.get("epochs", 200)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        best_valid_l2 = np.inf

        for epoch in range(epochs):
            self.train()
            train_l2 = 0
            for batch in train_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                out = self(x)

                l2 = self.loss_func(out, y)
                l2.backward() # use the l2 relative loss

                optimizer.step()
                train_l2 += l2.item()
                
            train_l2 /= len(train_loader.dataset)
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

            # wandb.log({"train_loss": train_l2, "valid_loss": valid_l2})
            print(f"Epoch {epoch}: Train loss={train_l2:.6f}, Validation loss={valid_l2:.6f} {saved}")
            
        self.load_state_dict(best_model_state_dict)
        train_l2 = self.test(train_loader, **fit_params)["loss"]
        if valid_loader is not None:
            valid_l2 = self.test(valid_loader, **fit_params)["loss"]
        else:
            valid_l2 = train_l2
        print(f"Finished training with best train loss: {train_l2:.6f} and validation loss: {valid_l2:.6f}")

    def finetune(self, finetune_loader, **fit_params):
        lr = fit_params.get("lr", 1e-3)
        epochs = fit_params.get("epochs", 100)

        # Fine tune the last n parameters. 
        # For example, finetuning only last linear layer, value is 2.
        last_n_parameters = fit_params.get("last_n_parameters", 2)
        finetune_parameters = list(self.parameters())[-last_n_parameters:]
        optimizer = torch.optim.Adam(finetune_parameters, lr=lr, weight_decay=1e-4)

        for epoch in range(epochs):
            self.train()
            train_l2 = 0
            for batch in finetune_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                
                optimizer.zero_grad()
                out = self(x)

                l2 = self.loss_func(out, y)
                l2.backward() # use the l2 relative loss

                optimizer.step()
                train_l2 += l2.item()
                
            train_l2 /= len(finetune_loader.dataset)

            print(f"Epoch {epoch}: {train_l2:.6f}")
            
        train_l2 = self.test(finetune_loader, **fit_params)["loss"]
        print(f"Finished finetuning with loss: {train_l2:.6f}")

    def test(self, test_loader, **test_params):
        self.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                out = self(x)
                test_l2 += self.loss_func(out, y).item()

        test_l2 /= len(test_loader.dataset)

        return {"loss": test_l2}

