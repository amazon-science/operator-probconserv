import numpy as np
import torch
import random
import json
from hashlib import sha1
from einops import reduce
from collections import defaultdict
import itertools
import time

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super().__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        if type(x) == tuple:
            x = x[0]
        return self.rel(x, y)
    

# normalization, pointwise gaussian
class UnitGaussianNormalizer:
    def __init__(self, x, eps=0.00001, reduce_dim=[0], verbose=True):
        super().__init__()
        n_samples, *shape = x.shape
        self.sample_shape = shape
        self.verbose = verbose
        self.reduce_dim = reduce_dim

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, reduce_dim, keepdim=True).squeeze(0)
        self.std = torch.std(x, reduce_dim, keepdim=True).squeeze(0)
        self.eps = eps
        
        if verbose:
            print(f'UnitGaussianNormalizer init on {n_samples}, reducing over {reduce_dim}, samples of shape {shape}.')
            print(f'   Mean and std of shape {self.mean.shape}, eps={eps}')

    def encode(self, x):
        # x = x.view(-1, *self.sample_shape)
        x = x - self.mean
        x = x / (self.std + self.eps)
        # x = (x.view(-1, *self.sample_shape) - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x.view(self.sample_shape) * std) + mean
        # x = x.view(-1, *self.sample_shape)
        x = x * std
        x = x + mean

        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def nll_mu_var(out, y):
    # out: Tuple of tensors (mu, var, ...)
    mu, var = out[0], out[1]
    nll = ((mu - y).pow(2)/var + torch.log(var)).sum()
    return nll


# Metrics
def compute_mse_by_t(mu, y, reduce="sum"):
    # out & y: nf nx nt d
    test_mse_by_t = ((mu - y)**2).sum(dim=[0, 1, 3]) / y.shape[1]
    if reduce == "mean":
        # Mean over n_samples
        test_mse_by_t /= y.shape[0]
    return test_mse_by_t

def compute_mse_by_example(mu, var, y):
    # out & y: nf nx nt d
    test_mse_by_example = ((mu - y)**2).mean(dim=[1, 2, 3])
    return test_mse_by_example

def compute_nll_by_example(mu, var, y):
    nll_by_example = ((mu - y).pow(2)/var + torch.log(2 * np.pi * var)).sum(dim=[1,2,3]) / 2
    return nll_by_example

def compute_nMeRCI(mu, var, y, alpha=0.95):
    # Compute n-MeRCI (normalized Mean Rescaled Confidence Interval) for correlation between uncertainty and errors.
    # Papers: https://arxiv.org/pdf/1908.07253.pdf, https://www.sciencedirect.com/science/article/pii/S0045782522004595#b55
    # Smaller values (closer to zero) is better.

    mae = torch.abs(mu - y).sum(dim=[1,2,3])
    std = torch.sqrt(var.sum(dim=[1,2,3]))
    lamda = mae / std
    lamda_alpha = torch.quantile(lamda, alpha)

    # Should be equal to alpha
    # print((mae <= std * lamda_alpha).float().mean())

    num = (lamda_alpha * std).mean() - mae.mean()
    denom = mae.max() - mae.mean()
    return num/denom


def compute_rmsce(mu, var, y, nbins=10):
    # Compute root mean squared calibration error.
    dist = torch.distributions.Normal(mu, torch.sqrt(var)+1e-10)
    ps = torch.linspace(0, 1, nbins+1)
    calibration_err = [(p - (y <= dist.icdf(p)).float().mean(dim=0))**2 for p in ps] 
    calibration_err = torch.stack(calibration_err).mean(dim=0).sqrt()
    return calibration_err.mean()

def compute_crps_by_example(mu, var, y, nbins=10):
    # Compute Continuous Ranked Probability Score (CRPS)
    # (https://www.jstor.org/stable/23243806?seq=4, https://arxiv.org/pdf/2102.00968.pdf)
    ps = torch.linspace(0, 1, nbins+1)[1:-1]
    dist = torch.distributions.Normal(mu, torch.sqrt(var)+1e-10)
    crps = 0.
    for p in ps:
        y_pred_at_p = dist.icdf(p)
        ql_p = ((y_pred_at_p > y).int() - p) * (y_pred_at_p - y)
        crps += ql_p
    crps *= 2/len(ps)
    return crps.mean(dim=[1,2,3])

def compute_piw_by_example(mu, var, y):
    # Assumes p=0.95
    std = torch.sqrt(var)
    piw = 2 * 1.96 * std 
    return piw.mean(dim=[1,2,3])

def compute_forward_time(model, x, repetitions=100):
    warmup = repetitions // 10
    times = []
    for i in range(warmup + repetitions):
        t0 = time.time()
        _ = model(x)
        torch.cuda.current_stream().synchronize()
        time_taken = (time.time() - t0) * 1000   # in ms
        if i >= warmup:
            times.append(time_taken)
    return np.mean(times)

def compute_n_params(model):
    n_params = 0
    for p in model.parameters():
        n_params += np.prod(p.shape)
    return int(n_params)

def compute_n_flops(model_name, Np, fno_modes, fno_width, n_layers, n_models):
    # Assumes d_i = d_o = 1
    lifting_layer = 2 * Np * fno_width
    fourier_layer = 10 * fno_width * Np * np.log2(Np) + fno_modes * (2 * fno_width**2 - fno_width) + 2 * Np * fno_width**2
    projection_layer = 2 * Np * fno_width

    if model_name.lower() == 'EnsembleFNO2d'.lower():
        n_flops = n_models * (lifting_layer + n_layers * fourier_layer + projection_layer)
    elif model_name.lower() == 'DiverseFNO2d'.lower():
        n_flops = lifting_layer + n_layers * fourier_layer + n_models * projection_layer
    else:
        n_flops = -1

    return int(n_flops)

def compute_all_metrics(out, y, results, metrics=None):
    if type(out) == tuple:
        mu, var = out[0], out[1]
    else:
        mu = out
        var = torch.zeros_like(mu) + 1e-20
    
    if metrics is None:
        metrics = ["mse", "nll", "piw", "crps"]

    results_ = {}

    for metric in metrics:
        metric_fn = globals()[f"compute_{metric}_by_example"]
        results_[f"{metric}_by_example"] = metric_fn(mu, var, y).detach().cpu()
        results_[metric] = results_[f"{metric}_by_example"].sum().item()

    for key in results_.keys():
        if key not in results:
            results[key] = results_[key]
        else:
            if key.endswith("by_example"):
                results[key] = torch.cat([results[key], results_[key]], dim=0)
            else:
                results[key] += results_[key]
    return results


def plot_at_time(index, t_plot, t_sliced, grid, y, mu, std, ax, **kwargs):
    ax.plot(grid, y[index, :, t_plot], label=f"True Solution (t={t_sliced[t_plot]:.1f})")
    ax.plot(grid, mu[index, :, t_plot], label=f"Predicted (t={t_sliced[t_plot]:.1f})")
    if std is not None:
        ax.fill_between(grid, mu[index, :, t_plot]-3*std[index, :, t_plot], mu[index, :, t_plot]+3*std[index, :, t_plot], color='b', alpha=0.1)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x, t)$")
    ax.legend()
    
    if "ylim" in kwargs:
        ax.set_ylim(*kwargs["ylim"])
    if "title" in kwargs:
        ax[0].set_title(kwargs["title"])
    # ax[0].set_xlim(-0.05, 1.05)
    # ax[0].set_ylim(-1, 1)

def set_seed(seed=314_271):
    # Set seed for random, numpy and torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=True

def config_to_hash(config):
    config_repr = json.dumps(config, sort_keys=True)
    return sha1(config_repr.encode()).hexdigest()

def dict_to_file(d, filepath):
    with open(filepath, 'w') as f:
        json.dump(d, f)

def filter_config(config, keys, mode="remove", new_config=None):
    if mode == "remove":
        if new_config is None:
            new_config = config.copy()
        new_config_keys = list(new_config.keys())
        for key in new_config_keys:
            if key in keys or "." in key:
                new_config.pop(key)
    elif mode == "add":
        if new_config is None:
            new_config = {}
        for key in keys:
            new_config[key] = config[key]

    return new_config


def generate_commands(filename, datasets, models, other_config, seed=0):
    if type(seed) == int:
        seed = [seed]

    commands = []

    for s in seed:
        for dataset_name, dataset_config in datasets.items():
            for model_name, model_config in models.items():
                dataset_name = dataset_name.split(":")[0]
                model_name = model_name.split(":")[0]

                command = f"python -u {filename} "
                command += f"--model={model_name} "
                command += f"--dataset={dataset_name} "
                command += f"--seed={s} "

                # Dataset & Model parameters
                config = dataset_config | model_config | other_config
                values = [[f"{k}" if vi=="" else f"{k}={vi}" for vi in v] for k,v in config.items()]
                for p in itertools.product(*values):
                    commands.append(command + " ".join(p))

    return commands
