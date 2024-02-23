import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import utils
from copy import deepcopy

# This code is borrowed from FNO git repository: https://github.com/zongyi-li/fourier_neural_operator

device = "cuda" if torch.cuda.is_available() else "cpu" 

################################################################
#  1d fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x
    
    
    
class FNO1d(nn.Module):
    def __init__(self, 
        modes, 
        width,
        input_dim=1,
        lb=0,
        ub=1,
        last_layer_reshape=False):
        super().__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.lb = lb # lower value of the domain
        self.ub = ub # upper value of the domain
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(input_dim+1, self.width) # input channel is 2: (a(x), x)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.last_layer_reshape = last_layer_reshape
        self.loss_func = utils.LpLoss(size_average=False)

    def forward(self, x, grid=None):
        """
        Forward function of the Neural Operator.
        
        Args:
            x: array representing the input of the Neural Operator, 
                given by the initial condition of the PDE
        """
        
        if grid:
            grid = self.get_given_grid(grid, x.shape, x.device)
        else:
            grid = self.get_grid(x.shape, x.device)
        
        x = torch.cat((x, grid), dim=-1)     
        x = self.fc0(x)
        
        x = x.permute(0, 2, 1)
                
        x1 = self.conv0(x) 
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        x1 = self.conv1(x) 
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        
        x1 = self.conv2(x) 
        x2 = self.w2(x)
        x = F.gelu(x1 + x2)
        
        x1 = self.conv3(x) 
        x2 = self.w3(x)
        
        x = x.permute(0, 2, 1)
        
        x = self.fc1(x)
        x = F.gelu(x)
        
        # Last layer reshape for Laplace approximation
        if self.last_layer_reshape:
            x = rearrange(x, 'b p d -> (b p) d')

        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(self.lb, self.ub, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    def get_given_grid(self, grid, shape, device):
        batchsize, size_x = shape[0], shape[1]
        grid = grid.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return grid.to(device)
    
    def fit(self, train_loader, valid_loader, grid_train=None, **fit_params):
        batch_size = fit_params.get("batch_size", 20)
        lr = fit_params.get("lr", 1e-3)
        step_size = fit_params.get("step_size", 1e-3)
        gamma = fit_params.get("gamma", 1e-3)
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
                out = self(x, grid_train)

                l2 = self.loss_func(out.view(batch_size, -1), y.view(batch_size, -1))
                l2.backward() # use the l2 relative loss

                optimizer.step()
                train_l2 += l2.item()
                
            train_l2 /= len(train_loader.dataset)
            scheduler.step()
            valid_l2 = self.test(valid_loader, grid_train, **fit_params)

            saved = "" 
            if valid_l2 < best_valid_l2:
                best_valid_l2 = valid_l2
                best_model_state_dict = deepcopy(self.state_dict())
                saved = "(saved)"

            print(f"Epoch {epoch}: {train_l2:.6f}, {valid_l2:.6f} {saved}")
            
        self.load_state_dict(best_model_state_dict)
        train_l2 = self.test(train_loader, grid_train, **fit_params)
        valid_l2 = self.test(valid_loader, grid_train, **fit_params)
        print(f"Finished training with best train loss: {train_l2:.6f} and validation loss: {valid_l2:.6f}")

    def test(self, test_loader, grid=None, **test_params):
        batch_size = test_params.get("batch_size", 20)
        self.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)

                out = self(x, grid)

                test_l2 += self.loss_func(out.view(batch_size, -1), y.view(batch_size, -1)).item()
        test_l2 /= len(test_loader.dataset)
        return test_l2



