"""
GraN-DAG
Copyright © 2019 Sébastien Lachapelle, Philippe Brouillard, Tristan Deleu
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from causalscbench.third_party.dcdfg.dcdfg.utils.dag_optim import bisect, is_acyclic


class LinearModularGaussianModule(nn.Module):
    def __init__(self, num_vars, num_modules, constraint_mode="matrix_power"):
        """
        Simplification for the "perfect known" context and the MLP framework
        :param int num_vars: number of variables in the system
        :param str constraint_mode: which constraint to use
        """
        super(LinearModularGaussianModule, self).__init__()
        self.num_vars = num_vars
        self.constraint_mode = constraint_mode
        self.num_modules = num_modules
        self.weights_U = nn.Parameter(data=torch.randn(self.num_vars, self.num_modules))
        self.weights_V = nn.Parameter(data=torch.randn(self.num_modules, self.num_vars))
        # initialize current adjacency matrix
        self.register_buffer(
            "weight_mask",
            torch.ones((self.num_vars, self.num_vars)) - torch.eye(self.num_vars),
        )
        self.biases = nn.Parameter(data=torch.randn(self.num_vars))
        self.log_stds = nn.Parameter(data=torch.zeros((self.num_vars,)))

        # Initialization for spectral radius constraint
        w_adj = self.get_w_adj()
        self.register_buffer("u", torch.zeros(w_adj.shape[0]))
        self.register_buffer("v", torch.zeros(w_adj.shape[0]))
        a, b = -3, 3
        with torch.no_grad():
            nn.init.trunc_normal_(self.u, a=a, b=b)
            nn.init.trunc_normal_(self.v, a=a, b=b)

        # get scaling factor and normalization factor for penalty
        with torch.no_grad():
            mat = torch.abs(w_adj)
            self.base_radius = self.spectral_radius_adj(mat, n_iter=100)
            self.constraint_norm = self.compute_dag_constraint(mat).item()
            if np.isinf(self.constraint_norm):
                raise ValueError("Error: constraint normalization is infinite")

    def compute_dag_constraint(self, w_adj):
        """
        Compute the DAG constraint on weighted adjacency matrix w_adj
        :param np.ndarray w_adj: the weighted adjacency matrix (each entry in [0,1])
        """
        if self.constraint_mode == "exp":
            return self.compute_dag_constraint_exp(w_adj)
        elif self.constraint_mode == "spectral_radius":
            return self.spectral_radius_adj(w_adj)
        elif self.constraint_mode == "matrix_power":
            return self.compute_dag_constraint_power(w_adj)
        else:
            raise ValueError(
                "constraint_mode needs to be in ['native_exp', 'scipy_exp', 'spectral_radius', 'matrix_power']."
            )

    def spectral_radius_adj(self, w_adj, n_iter=5):
        """
        Compute the spectral norm of w_adj with a power iteration.
        :param np.ndarray w_adj: the weighted adjacency matrix (each entry in [0,1])
        """
        with torch.no_grad():
            for _ in range(n_iter):
                self.v = F.normalize(w_adj.T @ self.v, dim=0)
                self.u = F.normalize(w_adj @ self.u, dim=0)
        return self.v.T @ w_adj @ self.u / (self.v @ self.u)

    def compute_dag_constraint_power(self, w_adj):
        """
        Compute the DAG constraint DIBS style via a matrix power.
        :param np.ndarray w_adj: the weighted adjacency matrix (each entry in [0,1])
        """
        d = w_adj.shape[0]
        eye = torch.eye(w_adj.shape[0], device=w_adj.device)
        return (
            torch.trace(
                torch.linalg.matrix_power(eye + w_adj / (d * self.base_radius), d)
            )
            - d
        )

    def compute_dag_constraint_exp(self, w_adj):
        return torch.trace(torch.matrix_exp(w_adj / self.base_radius)) - w_adj.shape[0]

    def forward(self, x):
        """
        :param x: batch_size x num_vars
        :return: batch_size x num_vars, the parameters of each variable conditional
        """
        x = (
            torch.matmul(
                x, self.weight_mask * torch.matmul(self.weights_U, self.weights_V)
            )
            + self.biases
        )
        return x

    def log_likelihood(self, x):
        """
        Return log-likelihood of the model for each example.
        WARNING: This is really a joint distribution only if the DAGness constraint on the mask is satisfied.
                 Otherwise the joint does not integrate to one.
        :param x: (batch_size, num_vars)
        :return: (batch_size, num_vars) log-likelihoods
        """
        density_params = self.forward(x)
        stds = torch.sqrt(torch.exp(self.log_stds) + 1e-4)
        return torch.distributions.Normal(density_params, stds.unsqueeze(0)).log_prob(x)

    def threshold(self):
        pos_adj = torch.abs(self.get_w_adj()).cpu().detach().numpy()

        def acyc(t):
            return float(is_acyclic(pos_adj >= t)) - 0.5

        threshold = bisect(acyc, 0, 5)
        assert acyc(threshold) > 0

        pred_adj = pos_adj >= threshold
        print(f"threshold:{threshold}")
        print(f"numel:{pred_adj.sum()}")
        # force weight mask to be the DAG
        self.weight_mask.copy_(torch.tensor(pred_adj, device=self.weight_mask.device))

    def losses(self, x, mask):
        """
        Compute the loss. If intervention is perfect and known, remove
        the intervened targets from the loss with a mask.
        """
        log_likelihood = torch.sum(self.log_likelihood(x) * mask, dim=0) / mask.size(0)
        # constraint related, square as values could be negative
        w_adj = torch.abs(self.get_w_adj())
        h = self.compute_dag_constraint(w_adj) / self.constraint_norm
        reg = torch.abs(self.weights_U).sum() + torch.abs(self.weights_V).sum()
        reg = 0.5 * reg / (self.weights_U.shape[0] * self.weights_V.shape[0])
        losses = (-torch.mean(log_likelihood), h, reg)
        return losses

    def check_acyclicity(self):
        to_keep = torch.abs(self.get_w_adj()) > 0.3
        return is_acyclic(to_keep.cpu().numpy())

    def get_w_adj(self):
        """Get weighted adjacency matrix"""
        return self.weight_mask * torch.matmul(self.weights_U, self.weights_V)
