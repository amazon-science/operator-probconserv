import numpy as np
import torch
from torch.distributions import Normal
from einops import rearrange, reduce, repeat

# This code is borrowed from ProbConserv git repository: https://github.com/amazon-science/probconserv

def meshgrid(ts, xs):
    _, nt = ts.shape
    _, nx = xs.shape
    ts = repeat(ts, "nf nt -> nf nt nx", nx=nx)
    xs = repeat(xs, "nf nx -> nf nt nx", nt=nt)
    return torch.stack((ts, xs), dim=-1)

def _get_riemman_delta(x):
    x_diff = torch.diff(x, dim=2)
    assert torch.all(x_diff >= 0)
    zero_pad_shape = (*x.shape[:2], 1)
    zero_pad = torch.zeros(*zero_pad_shape, device=x.device)
    x_delta_l = torch.cat((x_diff, zero_pad), dim=2)
    x_delta_r = torch.cat((zero_pad, x_diff), dim=2)
        
    riemann_type = "trapezoid"
    if riemann_type == "trapezoid":
        x_delta = 0.5 * (x_delta_l + x_delta_r)
    elif riemann_type == "rhs":
        x_delta = x_delta_r
    else:
        return NotImplementedError()
    return x_delta


def _apply_constraint(target_y_dist, target_inputs, mass_rhs, precis_g=np.inf, second_deriv_alpha=None, use_double_on_constraint=True):
    # target_inputs: nf nt nx 2
    # target_outputs: nf (nt nx) 1
    # mass_rhs: nf nt
    nf, nt, nx, _ = target_inputs.shape

    mu = rearrange(target_y_dist.loc, "nf (nt nx) 1 -> nf nt nx 1", nt=nt, nx=nx)
    masses_at_t = rearrange(mass_rhs, "nf nt -> nf nt 1 1")

    input_grid = rearrange(target_inputs, "nf nt nx d -> nf nt nx d", nt=nt, nx=nx)
    x = input_grid[:, :, :, 1]

    x_delta = _get_riemman_delta(x)

    g = rearrange(x_delta, "nf nt nx -> nf nt 1 nx")

    precis_g = torch.tensor(precis_g)
    precis_g = precis_g.reshape(1, 1)
    precis_g = rearrange(precis_g, "nf nt -> nf nt 1 1")

    eye = torch.eye(nx, device=g.device)
    eye = rearrange(eye, "nx1 nx2 -> 1 1 nx1 nx2")
    cov = target_y_dist.scale.pow(2)
    cov = rearrange(cov, "nf (nt nx) 1 -> nf nt nx 1", nt=nt)
    
    if second_deriv_alpha is not None:
        g2 = _get_second_deriv_mat(nx).to(g.device)
        g2 = rearrange(g2, "nxm2 nx -> 1 1 nxm2 nx")
        var_g2 = _get_second_derivative_var(cov, alpha=second_deriv_alpha).to(g.device)
        b = torch.zeros(1, 1, device=g2.device)
        mu, cov_mat = _apply_g(g2, var_g2, cov, mu, b)
    else:
        cov_mat = cov * eye

    var_g = 1 / precis_g
    
    if use_double_on_constraint:
        g = g.double()
        var_g = var_g.double()
        cov_mat = cov_mat.double()
        mu = mu.double()
        masses_at_t = masses_at_t.double()

    n_g = g.size(2)
    eye_g = torch.ones(1, 1, n_g, n_g, device=g.device, dtype=g.dtype)
    
    g_times_cov = g.matmul(cov_mat)
    gtr = g.transpose(3, 2)
    small_a = eye_g * var_g + (g_times_cov.matmul(gtr))
    rinv1 = torch.linalg.solve(small_a, g_times_cov)
    gtr_rinv1 = gtr.matmul(rinv1)
    new_cov = cov_mat.matmul(eye - gtr_rinv1)
    rinv2 = torch.linalg.solve(small_a, g.matmul(mu) - masses_at_t)
    new_mu = mu - cov_mat.matmul(gtr.matmul(rinv2))

    return new_mu.float(), new_cov.float()

def get_empirical_mass_rhs(outs):
    # outs: nf nx nt
    return 0.5 * (
        reduce(outs[:, 1:], "nf nx nt -> nf nt", "mean")
        + reduce(outs[:, :-1], "nf nx nt -> nf nt", "mean")
    )

def apply_constraint(mu, std, mass_rhs_func, t, tpred, grid_train, precis_g=np.inf, second_deriv_alpha=None):
    old_mu = rearrange(mu, "nf nx nt -> nf (nt nx) 1")
    old_std = rearrange(std, "nf nx nt -> nf (nt nx) 1")
    dist = Normal(old_mu, old_std)
    
    t_sliced = t[slice(*tpred)]
    ts = repeat(t_sliced, "nt -> nf nt", nf=mu.shape[0])
    xs = repeat(grid_train, "nx -> nf nx", nf=mu.shape[0])
    inputs = meshgrid(ts, xs)

    # Change
    # mass_rhs = torch.zeros(mu.shape[0], t_sliced.shape[0])
    mass_rhs = mass_rhs_func(inputs)
    
    new_mu, new_cov = _apply_constraint(dist, inputs, mass_rhs, precis_g=precis_g, second_deriv_alpha=second_deriv_alpha, use_double_on_constraint=True)
    new_std = torch.sqrt(torch.diagonal(new_cov, dim1=2, dim2=3))

    new_mu = rearrange(new_mu.squeeze(-1), "nf nt nx -> nf nx nt")
    new_std = rearrange(new_std, "nf nt nx -> nf nx nt")
    return new_mu, new_std, new_cov, mass_rhs

def plot_conservation(index, mu, mass_rhs, ax, **kwargs):
    label = kwargs.get("label", None)
    val = get_empirical_mass_rhs(mu) - mass_rhs
    ax.plot(val[index], label=label)
    ax.set_ylabel("Conservation Error")
    ax.set_xlabel("t")
    ax.legend()

def get_empirical_mass_rhs(outs):
    # outs: nf nx nt
    return 0.5 * (
        reduce(outs[:, 1:], "nf nx nt -> nf nt", "mean")
        + reduce(outs[:, :-1], "nf nx nt -> nf nt", "mean")
    )

def _get_second_deriv_mat(nx):
    eye = torch.eye(nx)
    eye1 = eye[:-2]
    eye2 = eye[1:-1] * -2
    eye3 = eye[2:]
    return eye1 + eye2 + eye3


def _get_second_deriv_mat_autocor(nx, alpha=0.5):
    eye = torch.eye(nx)
    eye1 = eye[:-2] + ((alpha - 2) * alpha)
    eye2 = eye[1:-1] * -2 + alpha
    eye3 = eye[2:]
    return eye1 + eye2 + eye3


def _get_second_derivative_var(cov, alpha=0.5):
    nf, nt, nx, _ = cov.shape
    cov0 = cov[:, :, :-2]
    cov1 = cov[:, :, 1:-1]
    cov2 = cov[:, :, 2:]

    return (
        cov0
        + 4 * cov1
        + cov2
        - 4 * alpha * cov0.sqrt() * cov1.sqrt()
        + 2 * (alpha**2) * cov0.sqrt() * cov2.sqrt()
        - 4 * alpha * cov1.sqrt() * cov2.sqrt()
    )

def _apply_g(g, var_g, cov, mu, mass_rhs):  # noqa: WPS210
    _, _, nx, _ = mu.shape
    _, _, ng, _ = g.shape
    eye = torch.eye(nx, device=g.device)
    eye = rearrange(eye, "nx1 nx2 -> 1 1 nx1 nx2")
    eye_g = torch.eye(ng, device=g.device)
    eye_g = rearrange(eye_g, "ng1 ng2 -> 1 1 ng1 ng2")
    gtr = g.transpose(3, 2)
    small_a = eye_g * var_g + (g.matmul(cov * gtr))
    rinv1 = torch.linalg.solve(small_a, g.matmul(cov * eye))
    new_cov = cov * (eye - gtr.matmul(rinv1))

    b = mass_rhs.unsqueeze(-1).unsqueeze(-1)
    rinv2 = torch.linalg.solve(small_a, g.matmul(mu) - b)
    new_mu = mu - cov * gtr.matmul(rinv2)
    return new_mu, new_cov




