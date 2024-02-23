"""
Experiment: testing FNO models on OOD parameters

Usage:
experiment_ood_params.py --model=<model> --dataset=<dataset> [--dataset_params=<dataset_params>] [--train_ood_dataset_params=<params>] [--ood_dataset_params=<ood_dataset_params>] 
                        [options] 

Options:
--model=<model>                             Model: EnsembleFNO2d, DiverseFNO2d, OutputVarFNO2d, MCDropoutFNO2d
--dataset=<dataset>                         Dataset: HeatEquation_1D, PME_1D, StefanPME_1D, LinearAdvection_1D
--dataset_params=<dataset_params>           Dataset parameters, comma separated values param1_lb, param1_ub, param2_lb, param2_ub, ...
                                            (e.g., for heat equation, --dataset_params=1,5,0,0).
--ood_dataset_params=<ood_dataset_params>   OOD Dataset parameters (similar to --dataset_params).
--n_samples=<n_samples>                     Size of dataset. [default: 400]
--grid_len=<grid_len>                       Size of grid. [default: 100]
--time_len=<time_len>                       Number of time steps. [default: 100]
--predict_time=<predict_time>               Prediction at time slice. [default: 0,-1,5]
--no_train                                  Use already fit model 
--batch_size=<batch_size>                   Batch size [default: 20]
--fno_modes=<fno_modes>                     FNO modes[default: 12]
--fno_width=<fno_width>                     FNO width [default: 32]
--lr=<lr>                                   Learning rate [default: 1e-3]
--epochs=<epochs>                           Epochs [default: 500]
--tplot=<tplot>                             Plot time [default: 0.5]
--seed=<seed>                               Seed [default: 314271]
--m.n_models=<n_models>                     Number of models or number of inferences [default: 10]
--m.reg_type=<reg_type>                     Regularization type [default: ""]
--m.reg_strength=<reg_strength>             Regularization strength [default: 0]
--m.drop_prob=<drop_prob>                   Dropping probability [default: 0.1]
--train_ood_dataset_params=<params>         OOD Dataset parameters for training using only inputs (no targets available)
--m.n_regularize=<n_regularize>             Number of heads to sample each epoch to enforce diversity regularization [default: 5]
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from models.FNO2d import FNO2d
from models.DiverseFNO2d import DiverseFNO2d
from models.UncertainNO import *
import utils
from einops import rearrange, reduce, repeat
import os
from docopt import docopt
import dill
from datasets import *
import probconserv
import sys

args = docopt(__doc__)

device = "cuda" if torch.cuda.is_available() else "cpu" 
experiment_name = "trial"
print(f"Experiment: {experiment_name}")
print(args)
save_args = utils.filter_config(args, ["generate", "--no_train", "--ood_dataset_params", "--tplot"], mode="remove")  # Also removes "." keys

is_train = not bool(args["--no_train"])

# Parameters
n_x = int(args["--grid_len"])
n_t = int(args["--time_len"])
n_samples = int(args["--n_samples"])
n_train = int(0.8 * n_samples)
n_valid = int(0.2 * n_samples)
n_test = n_samples // 2

is_markov = False

dataset = args["--dataset"]
dataset_params = [float(val) for val in args["--dataset_params"].split(",")]
train_ood_dataset_params = [float(val) for val in args["--train_ood_dataset_params"].split(",")]
ood_dataset_params = train_ood_dataset_params
if not is_train:
    ood_dataset_params = [float(val) for val in args["--ood_dataset_params"].split(",")]

tpred = [int(val) for val in args["--predict_time"].split(",")]

fno_modes = int(args["--fno_modes"])
fno_width = int(args["--fno_width"])

batch_size = int(args["--batch_size"])
lr = float(args["--lr"])
epochs = int(args["--epochs"])
step_size = 50
gamma = 0.5
# ################

# Set seed
utils.set_seed(int(args["--seed"]))

# Generate dataset
if dataset.lower() == "HeatEquation_1D".lower():
    t = torch.linspace(0, 1, n_t)
    grid = torch.linspace(0, 2 * np.pi, n_x)
    dataset_class = HeatEquation_1D
elif dataset.lower() == "PME_1D".lower():
    t = torch.linspace(0, 1, n_t)
    grid = torch.linspace(0, 1, n_x)
    dataset_class = PME_1D
elif dataset.lower() == "StefanPME_1D".lower():
    t = torch.linspace(0, 1, n_t)
    grid = torch.linspace(0, 1, n_x)
    dataset_class = StefanPME_1D
elif dataset.lower() == "LinearAdvection_1D".lower():
    t = torch.linspace(0, 1, n_t)
    grid = torch.linspace(0, 1, n_x)
    dataset_class = LinearAdvection_1D
else:
    raise NotImplementedError

t_sliced = t[slice(*tpred)]
T = len(t_sliced)

def get_xy_from_pu(p, u, is_markov=False):
    T = u.shape[2]
    if is_markov:
        x0, y0 = p, u
        
        y0_vectorized = rearrange(y0[:, :, 0:T-1], "nf nx nt 1 -> (nf nt) nx 1")
        x0 = repeat(x0, "nf nx 1 -> (nf nt) nx 1", nt=T-1)
        x = torch.cat([x0, y0_vectorized], dim=-1)
        
        y = rearrange(y0[:, :, 1:T], "nf nx nt 1 -> (nf nt) nx 1")
    else:
        x, y = p, u
        x = repeat(x, "nf nx 1 -> nf nx T 1", T=T)
    return x, y


if is_train:
    # Train data
    a, u, p = dataset_class.generate_dataset(n_train, grid, t, tpred, *dataset_params)
    x_train, y_train = get_xy_from_pu(p, u, is_markov=is_markov)

    # Validation data
    a, u, p = dataset_class.generate_dataset(n_valid, grid, t, tpred, *dataset_params)
    x_valid, y_valid = get_xy_from_pu(p, u, is_markov=is_markov)

    # In-distribution test data
    a, u, p = dataset_class.generate_dataset(n_test, grid, t, tpred, *dataset_params)
    x_id_test, y_id_test = get_xy_from_pu(p, u, is_markov=is_markov)

    # Out-of-distribution inputs only
    a, u, p = dataset_class.generate_dataset(n_test, grid, t, tpred, *train_ood_dataset_params)
    x_ood_test, y_ood_test = get_xy_from_pu(p, u, is_markov=is_markov)

    # Data loaders
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), 
                                            batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_valid, y_valid), 
                                            batch_size=batch_size, shuffle=False)
    id_test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_id_test, y_id_test), 
                                            batch_size=batch_size, shuffle=False)
    ood_test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_ood_test, y_ood_test), 
                                            batch_size=batch_size, shuffle=False)
else:
    # OOD test data
    a, u, p = dataset_class.generate_dataset(n_test, grid, t, tpred, *ood_dataset_params)
    x_ood_test, y_ood_test = get_xy_from_pu(p, u, is_markov=is_markov)
    ood_test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_ood_test, y_ood_test), 
                                            batch_size=batch_size, shuffle=False)


# Select model
uq = False
model_name = args["--model"]
n_models = 1
fno_modes2 = min(fno_modes, 12)
if args["--model"].lower() == "FNO2d".lower():
    FNO2d_params = {"modes1": fno_modes, "modes2": fno_modes2, "width": fno_width}
    model = FNO2d(**FNO2d_params).to(device)
elif args["--model"].lower().startswith("EnsembleFNO2d".lower()):
    FNO2d_params = {"modes1": fno_modes, "modes2": fno_modes2, "width": fno_width}
    n_models = int(args["--m.n_models"])
    utils.filter_config(args, ["--m.n_models"], mode="add", new_config=save_args)
    model = EnsembleNO(base_model_class=FNO2d, base_model_params=FNO2d_params, n_models=n_models)
    uq = True
elif args["--model"].lower().startswith("BayesianFNO2d".lower()):
    FNO2d_params = {"modes1": fno_modes, "modes2": fno_modes2, "width": fno_width}
    model = BayesianNO(base_model_class=FNO2d, base_model_params=FNO2d_params)
    uq = True
elif args["--model"].lower().startswith("MCDropoutFNO2d".lower()):
    FNO2d_params = {"modes1": fno_modes, "modes2": fno_modes2, "width": fno_width}
    dropout = float(args["--m.drop_prob"])
    n_dropouts = int(args["--m.n_models"])
    utils.filter_config(args, ["--m.n_models", "--m.drop_prob"], mode="add", new_config=save_args)
    model = MCDropoutNO(base_model_class=FNO2d, base_model_params=FNO2d_params, dropout=dropout, n_dropouts=n_dropouts)
    uq = True
elif args["--model"].lower().startswith("OutputVarFNO2d".lower()):
    FNO2d_params = {"modes1": fno_modes, "modes2": fno_modes2, "width": fno_width}
    model = OutputVarNO(base_model_class=FNO2d, base_model_params=FNO2d_params)
    uq = True
elif args["--model"].lower().startswith("DiverseFNO2d".lower()):
    FNO2d_params = {"modes1": fno_modes, "modes2": fno_modes2, "width": fno_width}
    lam = float(args["--m.reg_strength"])
    reg_type = args["--m.reg_type"]
    n_models = int(args["--m.n_models"])
    n_regularize = int(args["--m.n_regularize"])
    utils.filter_config(args, ["--m.n_models", "--m.reg_strength", "--m.reg_type", "--m.n_regularize"], mode="add", new_config=save_args)
    model = DiverseFNO2d(reg_loss=reg_type, n_outputs=n_models, bias_last=False, lam=lam, n_regularize=n_regularize, **FNO2d_params).to(device)
    uq = True
else:
    raise NotImplementedError

# Add ProbConserv or not
is_probconserv = True

# Folder to save results
root = '.'
config_hash = utils.config_to_hash(save_args)
run_folder = f"{root}/results/{experiment_name}/{args['--dataset']}_{args['--model']}_{config_hash}"
print(run_folder)
if is_train:
    os.makedirs(run_folder, exist_ok=True)
    utils.dict_to_file(save_args, os.path.join(run_folder, "config.json"))

model_path = os.path.join(run_folder, "model.pkl")
if is_train:
    x_ood_test = x_ood_test.to(device)
    model.fit(train_loader, valid_loader, x_test=x_ood_test, epochs=epochs, lr=lr, step_size=step_size, gamma=gamma)
    torch.save(model, model_path, pickle_module=dill)
else:
    model = torch.load(model_path, pickle_module=dill).to(device)

print("Model loaded")

def plot_and_save(prefix, idx, x, y, mu, std, new_mu=None, new_std=None, mass_rhs=None):
    ncol = 3 if is_probconserv else 1
    fig, ax = plt.subplots(1, ncol, figsize=(12//2*ncol, 9//2))
    ax = [ax] if ncol==1 else ax
    t_plot = min(int(float(args["--tplot"]) * len(t_sliced)), len(t_sliced)-1)
    error = (y[idx] - mu[idx]).abs()

    if std is None:
        std = np.zeros_like(mu)

    # Finding p% contiguous region with max std
    len_where_std_max = int(0.25 * y.shape[1])
    mx = np.argmax(np.convolve(std[idx, :, t_plot], np.ones(len_where_std_max), 'valid'))
    Mx = mx + len_where_std_max - 1
        
    utils.plot_at_time(idx, t_plot, t_sliced, grid, y, mu, std, ax[0])
    ax[0].set_title(f"{model_name}")
    ax[0].scatter(grid[mx], mu[idx, mx, t_plot], s=200, marker='|', c='red')
    ax[0].scatter(grid[Mx], mu[idx, Mx, t_plot], s=200, marker='|', c='red')

    if is_probconserv:
        utils.plot_at_time(idx, t_plot, t_sliced, grid, y, new_mu, new_std, ax[1], ylim=ax[0].get_ylim())
        ax[1].set_title(f"{model_name} + ProbConserv")
        probconserv.plot_conservation(idx, mu, mass_rhs, ax[2], label=f"{model_name}")
        probconserv.plot_conservation(idx, new_mu, mass_rhs, ax[2], label=f"+ProbConserv")
        ax[2].set_title("Conservation plot")
    
    param = x[idx, 0, 0] if len(x.shape)==3 else x[idx, 0, 0, 0]
    plt.suptitle(f"{prefix}, param={param:.2f}, MSE={(error**2).mean():.4f}")
    plt.savefig(f"{run_folder}/plot_{prefix}_tidx={t_plot}_{idx}.png", dpi=300)


def test(model, test_loader, **test_params):
    test_type = test_params.get("test_type", "id")
    mu = []
    var = []
    results = {}
    results["loss"] = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x, y = batch
            x, y = x.to(device), y.to(device)

            out = model(x)

            results["loss"] += model.loss_func(out, y).item()
            utils.compute_all_metrics(out, y, results)

            if uq:
                mu.append(out[0].detach().cpu())
                var.append(out[1].detach().cpu())
            else:
                mu.append(out.detach().cpu())

    for key in results.keys():
        if not key.endswith("by_example"):
            results[key] /= len(test_loader.dataset)
        if type(results[key]) == torch.Tensor:
            results[key] = results[key].tolist()

    # Plot
    mu = torch.cat(mu, dim=0)
    if uq:
        var = torch.cat(var, dim=0)
        std = torch.sqrt(var)
    else:
        var = None
        std = None
    x = test_loader.dataset.tensors[0]
    y = test_loader.dataset.tensors[1]

    if uq:
        results["nMeRCI_all"] = utils.compute_nMeRCI(mu, var, y).item()
        results["rmsce_all"] = utils.compute_rmsce(mu, var, y).item()

        if is_probconserv:
            mass_rhs_func = dataset_class.get_mass_rhs_func(x=x)
            new_mu, new_std, _, mass_rhs = probconserv.apply_constraint(
                mu=mu[:, :, :, 0], 
                std=std[:, :, :, 0], 
                mass_rhs_func=mass_rhs_func, 
                t=t, 
                tpred=tpred, 
                grid_train=grid, 
                precis_g=1e9,
                second_deriv_alpha=None,
            )
            new_mu = new_mu[:, :, :, None]
            new_std = new_std[:, :, :, None]
            new_var = new_std**2

            probconserv_results = utils.compute_all_metrics((new_mu, new_var), y, {})
            for key in probconserv_results.keys():
                if not key.endswith("by_example"):
                    probconserv_results[key] /= len(test_loader.dataset)
                if type(probconserv_results[key]) == torch.Tensor:
                    probconserv_results[key] = probconserv_results[key].tolist()

            probconserv_results["nMeRCI_all"] = utils.compute_nMeRCI(new_mu, new_var, y).item()
            probconserv_results["rmsce_all"] = utils.compute_rmsce(new_mu, new_var, y).item()

            cerr = (probconserv.get_empirical_mass_rhs(mu[:, :,  :, 0]) - mass_rhs).abs().sum(dim=-1)
            new_cerr = (probconserv.get_empirical_mass_rhs(new_mu[:, :, :, 0]) - mass_rhs).abs().sum(dim=-1)

            results["cerr_by_example"] = cerr.tolist()
            results["mcerr"] = cerr.mean().item()
            probconserv_results["cerr_by_example"] = new_cerr.tolist()
            probconserv_results["mcerr"] = new_cerr.mean().item()

            for key in probconserv_results.keys():
                results[f"pc.{key}"] = probconserv_results[key]
    
    # results["time"] = utils.compute_forward_time(model, x[:batch_size].to(device), repetitions=10)
    results["n_params"] = utils.compute_n_params(model)
    results["n_flops"] = utils.compute_n_flops(model_name, Np=n_x*n_t, fno_modes=fno_modes, fno_width=fno_width, n_layers=4, n_models=n_models)

    dataset_params_correct_type = dataset_params if test_type == "id" or test_type == "train" else ood_dataset_params

    mse_by_example = torch.tensor(results["mse_by_example"])
    random_idx = np.random.choice(mse_by_example.shape[0])
    _, worst_idx = mse_by_example.max(dim=0)
    _, best_idx = mse_by_example.min(dim=0)
    _, median_idx = mse_by_example.median(dim=0)

    for example_name, example_idx in zip(["random", "worst", "best", "median"], [random_idx, worst_idx, best_idx, median_idx]):
        if uq:
            results[f"examples.{example_name}"] = (mu[example_idx].tolist(), var[example_idx].tolist(), y[example_idx].tolist(), x[example_idx].tolist())
            if probconserv:
                results[f"pc.examples.{example_name}"] = (new_mu[example_idx].tolist(), new_var[example_idx].tolist(), y[example_idx].tolist(), x[example_idx].tolist())
        else:
            results[f"examples.{example_name}"] = (mu[example_idx].tolist(), None, y[example_idx].tolist(), x[example_idx].tolist())

        # prefix = f"{test_type}_{example_name}_params={dataset_params_correct_type}"
        # plot_and_save(prefix, example_idx, x.squeeze(-1), y.squeeze(-1), mu.squeeze(-1), std.squeeze(-1) if std is not None else None)

    utils.dict_to_file({"test_type": test_type, "params": dataset_params_correct_type, "results": results}, 
                       f"{run_folder}/results_{test_type}_params={dataset_params_correct_type}.json")

    return results

if is_train:
    train_loader_no_shuffle = torch.utils.data.DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=False)
    train_results = test(model, train_loader_no_shuffle, test_type="train")
    id_results = test(model, id_test_loader, test_type="id")

    print("In-domain results")
    print(f"MSE: {id_results['mse']}")
    print(f"n-MeRCI: {id_results['nMeRCI_all']}")
    print(f"RMSCE: {id_results['rmsce_all']}")

else:
    ood_results = test(model, ood_test_loader, test_type="ood")

    print("Out-of-domain results")
    print(f"MSE: {ood_results['mse']}")
    print(f"n-MeRCI: {ood_results['nMeRCI_all']}")
    print(f"RMSCE: {ood_results['rmsce_all']}")


