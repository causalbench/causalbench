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
import torch
import torch.nn as nn
import torch.nn.functional as F

from causalscbench.third_party.dcdfg.dcdfg.utils.dag_optim import GumbelAdjacency, is_acyclic


class MLPGaussianModule(nn.Module):
    def __init__(
        self,
        num_vars,
        num_layers,
        hid_dim,
        nonlin="leaky_relu",
        constraint_mode="matrix_power",
    ):
        """
        Simplification for the "perfect known" context and the MLP framework
        :param int num_vars: number of variables in the system
        :param int num_layers: number of hidden layers
        :param int hid_dim: number of hidden units per layer
        :param int num_params: number of parameters per conditional *outputted by MLP*
        :param str nonlin: which nonlinearity to use
        :param str constraint_mode: which constraint to use
        """
        super(MLPGaussianModule, self).__init__()
        self.num_vars = num_vars
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.nonlin = nonlin
        self.constraint_mode = constraint_mode

        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.log_stds = nn.Parameter(data=torch.zeros((self.num_vars,)))

        # initialize current adjacency matrix
        self.register_buffer(
            "adjacency",
            torch.ones((self.num_vars, self.num_vars)) - torch.eye(self.num_vars),
        )
        self.gumbel_adjacency = GumbelAdjacency(self.num_vars)

        self.zero_weights_ratio = 0.0
        self.numel_weights = 0

        # Instantiate the parameters of each layer in the model of each variable
        for i in range(self.num_layers + 1):
            in_dim = self.hid_dim
            out_dim = self.hid_dim

            # first layer
            if i == 0:
                in_dim = self.num_vars

            # last layer
            if i == self.num_layers:
                out_dim = 1

            self.weights.append(nn.Parameter(torch.zeros(num_vars, out_dim, in_dim)))
            self.biases.append(nn.Parameter(torch.zeros(num_vars, out_dim)))
            self.numel_weights += self.num_vars * out_dim * in_dim

        # init params
        with torch.no_grad():
            for node in range(self.num_vars):
                for i, w in enumerate(self.weights):
                    w = w[node]
                    nn.init.xavier_uniform_(
                        w, gain=nn.init.calculate_gain("leaky_relu")
                    )
                for i, b in enumerate(self.biases):
                    b = b[node]
                    b.zero_()

        # Initialization for spectral radius constraint
        if self.constraint_mode == "spectral_radius":
            w_adj = self.get_w_adj()
            self.register_buffer("u", torch.zeros(w_adj.shape[0]))
            self.register_buffer("v", torch.zeros(w_adj.shape[0]))
            a, b = -3, 3
            with torch.no_grad():
                nn.init.trunc_normal_(self.u, a=a, b=b)
                nn.init.trunc_normal_(self.v, a=a, b=b)
            # Pre-train power iteration variables
            self.spectral_radius_adj(w_adj, n_iter=100)

        # get normalization factor
        with torch.no_grad():
            full_adjacency = torch.ones((self.num_vars, self.num_vars)) - torch.eye(
                self.num_vars
            )
            self.constraint_norm = self.compute_dag_constraint(full_adjacency).item()

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
        return (
            torch.trace(
                torch.linalg.matrix_power(
                    torch.eye(w_adj.shape[0], device=w_adj.device) + w_adj / d, d
                )
            )
            - d
        )

    def compute_dag_constraint_exp(self, w_adj):
        return torch.trace(torch.matrix_exp(w_adj)) - w_adj.shape[0]

    def forward(self, x):
        """
        :param x: batch_size x num_vars
        :return: batch_size x num_vars * num_params, the parameters of each variable conditional
        """
        bs = x.size(0)
        num_zero_weights = 0

        for layer in range(self.num_layers + 1):
            # First layer, apply the mask
            if layer == 0:
                # sample the matrix M that will be applied as a mask at the MLP input
                M = self.gumbel_adjacency(bs)
                adj = self.adjacency.unsqueeze(0)
                x = (
                    torch.einsum("tij,bjt,ljt,bj->bti", self.weights[layer], M, adj, x)
                    + self.biases[layer]
                )
            # 2nd layer and more
            else:
                x = (
                    torch.einsum("tij,btj->bti", self.weights[layer], x)
                    + self.biases[layer]
                )

            # count number of zeros
            num_zero_weights += self.weights[layer].numel() - self.weights[
                layer
            ].nonzero().size(0)

            # apply non-linearity
            if layer != self.num_layers:
                x = F.leaky_relu(x) if self.nonlin == "leaky_relu" else torch.sigmoid(x)

        self.zero_weights_ratio = num_zero_weights / float(self.numel_weights)

        return torch.unbind(x, 1)

    def log_likelihood(self, x):
        """
        Return log-likelihood of the model for each example.
        WARNING: This is really a joint distribution only if the DAGness constraint on the mask is satisfied.
                 Otherwise the joint does not integrate to one.
        :param x: (batch_size, num_vars)
        :return: (batch_size, num_vars) log-likelihoods
        """
        density_params = self.forward(x)
        stds = torch.exp(self.log_stds)
        return torch.distributions.Normal(
            torch.cat(density_params, 1), stds.unsqueeze(0)
        ).log_prob(x)

    def losses(self, x, mask):
        """
        Compute the loss. If intervention is perfect and known, remove
        the intervened targets from the loss with a mask.
        """
        log_likelihood = torch.sum(self.log_likelihood(x) * mask, dim=0) / mask.size(0)
        # constraint related
        w_adj = self.get_w_adj()
        h = self.compute_dag_constraint(w_adj) / self.constraint_norm
        reg = torch.abs(w_adj).sum() / w_adj.shape[0] ** 2
        losses = (-torch.mean(log_likelihood), h, reg)
        return losses

    def threshold(self):
        # Final thresholding of all edges <= 0.5
        # and edges > 0.5 are set to 1
        with torch.no_grad():
            w_adj = self.get_w_adj()
            higher = (w_adj > 0.5).type_as(w_adj)
            lower = (w_adj <= 0.5).type_as(w_adj)
            self.gumbel_adjacency.log_alpha.copy_(higher * 100 + lower * -100)
            self.gumbel_adjacency.log_alpha.requires_grad = False
            self.adjacency.copy_(higher)

    def check_acyclicity(self):
        to_keep = (self.get_w_adj() > 0.5).type_as(self.adjacency)
        current_adj = self.adjacency * to_keep
        return is_acyclic(current_adj.cpu().numpy())

    def get_w_adj(self):
        """Get weighted adjacency matrix"""
        return self.gumbel_adjacency.get_proba() * self.adjacency

    def get_grad_norm(self, mode="wbx"):
        """
        Will get only parameters with requires_grad == True, simply get the .grad
        :param mode: w=weights, b=biases, x=extra_params (order is irrelevant)
        :return: corresponding dicts of parameters
        """
        grad_norm = 0

        if "w" in mode:
            for w in self.weights:
                grad_norm += torch.sum(w.grad**2)

        if "b" in mode:
            for b in self.biases:
                grad_norm += torch.sum(b.grad**2)

        if "x" in mode:
            for ep in self.extra_params:
                if ep.requires_grad:
                    grad_norm += torch.sum(ep.grad**2)

        return torch.sqrt(grad_norm)
