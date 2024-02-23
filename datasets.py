import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
import matplotlib.pyplot as plt
from einops import rearrange, repeat, reduce
from scipy.optimize import root_scalar
from scipy.special import erf
from functools import partial
from pathlib import Path
import scipy.io
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Resize
import os
import utils
                    

def meshgrid(ts, xs):
    _, nt = ts.shape
    _, nx = xs.shape
    ts = repeat(ts, "nf nt -> nf nt nx", nx=nx)
    xs = repeat(xs, "nf nx -> nf nt nx", nt=nt)
    return torch.stack((ts, xs), dim=-1)


class HeatEquation_1D:
    def __init__(self, nx_soln=1000):
        self.nx_soln = nx_soln  # Grid spacing to evaluate the solution (fixed)

    def _convection_diffusion_solution(self, x_start, t_values, nu, beta, source=0):
        nx = x_start.shape[0]
        forcing_term = np.zeros_like(x_start) + source  # G is the same size as u0

        ikx_pos = 1j * np.arange(0, nx / 2 + 1, 1)
        ikx_neg = 1j * np.arange(-nx / 2 + 1, 0, 1)
        ikx = np.concatenate((ikx_pos, ikx_neg))
        ikx2 = ikx * ikx

        uhat0 = np.fft.fft(x_start)
        nu_term = nu * ikx2 * t_values
        beta_term = beta * ikx * t_values
        nu_factor = np.exp(nu_term - beta_term)
        uhat = (
            uhat0 * nu_factor + np.fft.fft(forcing_term) * t_values
        )  # for constant, fft(p) dt = fft(p)*T
        return np.real(np.fft.ifft(uhat))

    def _convection_onedim_for_one_parameter(self, t_values, theta, nu):
        x_grid = np.linspace(0, 2*np.pi, self.nx_soln)
        x_start = np.sin(x_grid + theta)
        t_grid = repeat(t_values, "nt -> nt nx", nx=self.nx_soln)
        t_grid = t_grid.numpy()
        return self._convection_diffusion_solution(x_start, t_grid, nu, beta=0)

    def _convection_onedim(self, t_values, thetas, nus):
        n_function_draws = thetas.shape[0]
        u_list = []
        for i in range(n_function_draws):
            u_i = self._convection_onedim_for_one_parameter(
                t_values[i, :],
                thetas[i].item(),
                nus[i].item(),
            )
            u_list.append(u_i)
        u = np.stack(u_list, axis=0)
        return torch.from_numpy(u)

    def true_solution(self, inputs, thetas, nus):
        # inputs: [nf, nt, nx, d]
        nf, nt, _, _ = inputs.shape

        inputs = rearrange(inputs, "nf nt nx d -> nf (nt nx) d")
        ts = inputs[:, :, 0]
        xs = inputs[:, :, 1]
        
        xs = xs.unique(dim=1).reshape(nf, -1)
        ts = ts.unique(dim=1).reshape(nf, -1)
        
        tr_all = self._convection_onedim(ts, thetas, nus)

        grid = rearrange(inputs, "nf (nt nx) d -> nf nt nx d", nt=nt).clone()
        grid[:, :, :, 1] /= np.pi * 2
        grid = (grid - 0.5) * 2
        # (h w) to (x y)
        grid_x = grid[:, :, :, 1]
        grid_y = grid[:, :, :, 0]
        grid = torch.stack((grid_x, grid_y), dim=-1)

        tr = F.grid_sample(
            tr_all.unsqueeze(1).float(), grid, align_corners=True, mode="bilinear"
        ).squeeze(1)
        tr = rearrange(tr, "nf nt nx -> nf (nt nx) 1")
        return tr.float()

    @classmethod
    def get_mass_rhs_func(cls, x):
        def mass_rhs_func(inputs):
            nf, nt, _, _ = inputs.shape
            mass_rhs = torch.zeros(nf, nt)
            return mass_rhs
        return mass_rhs_func

    @classmethod
    def generate_dataset(cls, n_samples, grid, t, tpred=(-1, None, None), *dataset_params):
        nu_1, nu_2, theta_1, theta_2 = dataset_params
        thetas = theta_1 + (theta_2 - theta_1) * torch.rand(n_samples)
        conductivities = nu_1 + (nu_2 - nu_1) * torch.rand(n_samples)

        nt = t.shape[0]
        nx = grid.shape[0]

        ts = repeat(t, "nt -> nf nt", nf=n_samples)
        xs = repeat(grid, "nx -> nf nx", nf=n_samples)
        inputs = meshgrid(ts, xs)

        pde = HeatEquation_1D()
        outputs = pde.true_solution(inputs, thetas, conductivities)
        outputs = rearrange(outputs, "nf (nt nx) d -> nf nx nt d", nt=nt)

        params = repeat(conductivities, "nf -> nf nx 1", nx=nx)

        tpred = slice(*tpred)
        a, u = outputs[:, :, 0], outputs[:, :, tpred].squeeze(2)

        return a, u, params


class PME_1D:
    def __init__(self):
        pass

    def true_solution(self, inputs, degrees, scales):
        # inputs: [nf, nt, nx, d]
        nf, nt, _, _  = inputs.shape

        inputs = rearrange(inputs, "nf nt nx d -> nf (nt nx) d")
        ts = inputs[:, :, 0]
        xs = inputs[:, :, 1]

        degrees = rearrange(degrees, "nf -> nf 1")
        scales = rearrange(scales, "nf -> nf 1")

        xs = xs * scales
        
        us = degrees * F.relu(ts - xs)
        ys = us.pow(1 / degrees)
        return rearrange(ys, "nf nt_nx -> nf nt_nx 1")

    @classmethod
    def get_mass_rhs_func(cls, x):
        def mass_rhs_func(inputs):
            # nf, nt, _, _ = inputs.shape
            degrees = x[:, 0, 0]
            ts = inputs[:, :, 0, 0]
            a1 = 1 + (1 / degrees)
            mass_rhs = (degrees.pow(a1)) / (degrees + 1) * ts.pow(a1)
            return mass_rhs
        return mass_rhs_func
    
    def _mass_rhs(self, inputs, degrees):
        # inputs: [nf, nt, nx, d]
        degrees = rearrange(degrees, "nf -> nf 1")
        ts = inputs[:, :, 0, 0]
        a1 = 1 + (1 / degrees)
        return (degrees.pow(a1)) / (degrees + 1) * ts.pow(a1)

    @classmethod
    def generate_dataset(cls, n_samples, grid, t, tpred=(-1, None, None), *dataset_params):

        degree_1, degree_2 = dataset_params
        degrees = degree_1 + (degree_2 - degree_1) * torch.rand(n_samples)
        scales = torch.ones(n_samples)

        nt = t.shape[0]
        nx = grid.shape[0]

        ts = repeat(t, "nt -> nf nt", nf=n_samples)
        xs = repeat(grid, "nx -> nf nx", nf=n_samples)
        inputs = meshgrid(ts, xs)

        pde = cls()
        outputs = pde.true_solution(inputs, degrees, scales)
        outputs = rearrange(outputs, "nf (nt nx) d -> nf nx nt d", nt=nt)

        params = repeat(degrees, "nf -> nf nx 1", nx=nx)

        # a, u = outputs[:, :, 0], outputs[:, :, t.shape[0]//2]
        tpred = slice(*tpred)
        a, u = outputs[:, :, 0], outputs[:, :, tpred].squeeze(2)
        return a, u, params


class StefanPME_1D:
    def __init__(self):
        self.k_min = 0
        self.k_max = 1
            
    def true_solution(self, inputs, p_stars):
        # inputs: [nf, nt, nx, d]
        soln_list = []
        mass_rhs_list = []
        for input_, p_star in zip(inputs, p_stars):
            soln_i, mass_rhs_i = self._true_solution_one_parameter(input_, p_star)
            soln_list.append(soln_i)
            mass_rhs_list.append(mass_rhs_i)
        return torch.stack(soln_list, dim=0).unsqueeze(-1), torch.stack(mass_rhs_list)
        
    def _true_solution_one_parameter(self, inputs, p_star):
        # inputs: [nt, nx, d]
        ts_i = inputs[:, 0, 0]
        inputs = rearrange(inputs, "nt nx d -> (nt nx) d")
        ts, xs = np.split(inputs, 2, -1)
        
        _z1 = root_scalar(partial(StefanPME_1D._z1_objective, p_star=p_star), bracket=(0, 10)).root
        _alpha = 2 * np.sqrt(self.k_max) * _z1
        
        # c1
        num = 1 - p_star
        dem = erf(_alpha / (2 * (np.sqrt(self.k_max))))
        c1 = num / dem
        
        # c2
        num = p_star
        a = _alpha / (2 * np.sqrt(self.k_min))
        dem = 1 - erf(a)
        c2 = num / dem
    
        # p1
        a = xs / (2 * torch.sqrt(self.k_max * ts))
        p1 = 1 - c1 * erf(a)

        # p2
        if self.k_min == 0:
            p2 = torch.zeros_like(xs)
        else:
            a = xs / (2 * torch.sqrt(self.k_min * ts))
            p2 = c2 * (1 - erf(a))

        # mass_rhs
        mass_rhs = self._mass_rhs(ts_i, c1)
    
        x_star = _alpha * torch.sqrt(ts)
        p = p1 * (xs <= x_star) + p2 * (xs > x_star)
        p[np.isclose(xs, 0)] = 1.0
        return p.squeeze(-1), mass_rhs

    @classmethod
    def _z1_objective(cls, z1, p_star):
        a1 = p_star * erf(z1)
        a2 = z1 * np.exp(np.power(z1, 2))
        b = (1 - p_star) / np.sqrt(np.pi)
        return (a1 * a2) - b

    @classmethod
    def get_mass_rhs_func(cls, x):
        def mass_rhs_func(inputs):
            # Assumes self.kmax = 1
            # inputs: [nf, nt, nx, d]

            p_stars = x[:, 0, 0]
            mass_rhs = []
            for input_, p_star in zip(inputs, p_stars):
                ts_i = input_[:, 0, 0]
                _z1 = root_scalar(partial(StefanPME_1D._z1_objective, p_star=p_star), bracket=(0, 10)).root
                _alpha = 2 * _z1

                # c1
                num = 1 - p_star
                dem = erf(_alpha / 2)
                c1 = num / dem

                a1 = 2 * np.sqrt(1 / np.pi)
                mass_rhs_i = a1 * c1 * torch.sqrt(ts_i)
                mass_rhs.append(mass_rhs_i)
            
            mass_rhs = torch.stack(mass_rhs)
            return mass_rhs

        return mass_rhs_func
    
    def _mass_rhs(self, ts, c1):
        a1 = 2 * np.sqrt(self.k_max / np.pi)
        return a1 * c1 * torch.sqrt(ts)
    
    @classmethod
    def generate_dataset(cls, n_samples, grid, t, tpred=(-1, None, None), *dataset_params):
        p_star_1, p_star_2 = dataset_params
        p_stars = p_star_1 + (p_star_2 - p_star_1) * torch.rand(n_samples)

        nt = t.shape[0]
        nx = grid.shape[0]
        ts = repeat(t, "nt -> nf nt", nf=n_samples)
        xs = repeat(grid, "nx -> nf nx", nf=n_samples)
        inputs = meshgrid(ts, xs)

        pde = cls()
        outputs, mass_rhs = pde.true_solution(inputs, p_stars)
        outputs = rearrange(outputs, "nf (nt nx) d -> nf nx nt d", nt=nt)

        params = repeat(p_stars, "nf -> nf nx 1", nx=nx)

        tpred = slice(*tpred)
        a, u = outputs[:, :, 0], outputs[:, :, tpred].squeeze(2)
        # return a, u, params, mass_rhs[:, tpred]
        return a, u, params


class LinearAdvection_1D:
    def __init__(self):
        pass

    def true_solution(self, inputs, betas):
        # inputs: [nf, nt, nx, d]
        nf, nt, _, _  = inputs.shape
        
        inputs = rearrange(inputs, "nf nt nx d -> nf (nt nx) d")
        ts = inputs[:, :, 0]
        xs = inputs[:, :, 1]
        
        betas = rearrange(betas, "nf -> nf 1")
        ys = self.h(xs - ts * betas)
        return rearrange(ys, "nf nt_nx -> nf nt_nx 1")

    def h(self, x):
        return (x <= 0.5).float()

    @classmethod
    def get_mass_rhs_func(cls, x):
        def mass_rhs_func(inputs):
            # nf, nt, _, _ = inputs.shape
            betas = x[:, 0, 0]
            ts = inputs[:, :, 0, 0]
            mass_rhs = 0.5 + torch.minimum(betas * ts, torch.tensor(0.5))
            return mass_rhs
        return mass_rhs_func

    @classmethod
    def generate_dataset(cls, n_samples, grid, t, tpred=(-1, None, None), *dataset_params):
        beta_1, beta_2 = dataset_params
        betas = beta_1 + (beta_2 - beta_1) * torch.rand(n_samples)

        nt = t.shape[0]
        nx = grid.shape[0]

        ts = repeat(t, "nt -> nf nt", nf=n_samples)
        xs = repeat(grid, "nx -> nf nx", nf=n_samples)
        inputs = meshgrid(ts, xs)

        pde = LinearAdvection_1D()
        outputs = pde.true_solution(inputs, betas)
        outputs = rearrange(outputs, "nf (nt nx) d -> nf nx nt d", nt=nt)

        params = repeat(betas, "nf -> nf nx 1", nx=nx)

        tpred = slice(*tpred)
        a, u = outputs[:, :, 0], outputs[:, :, tpred].squeeze(2)
        return a, u, params

if __name__ == '__main__':
    # grid = torch.linspace(0, 1, 500)
    # t = torch.linspace(0, 1, 100)
    # pde = Burger_1D_Dir(nu=0.001)
    # a, u = pde.generate_samples(n_samples=10, u_l_range=(0.75, 0.85), grid=grid, t=t)


    # n_t = 100
    # n_x = 50
    # t = torch.linspace(0, 1, n_t)
    # grid = torch.linspace(0, 2 * np.pi, n_x)
    # dataset_params = (2, 2, 0, np.pi/8)
    # a, u = HeatEquation_1D.generate_dataset(10, grid, t, *dataset_params)


    # n_t = 100
    # n_x = 100
    # t = torch.linspace(0, 1, n_t)
    # grid = torch.linspace(0, 1, n_x)
    # dataset_params = (2, 3, 1, 1)
    # a, u, params = PME_1D.generate_dataset(10, grid, t, (None, None, 5),  *dataset_params)

    pass
