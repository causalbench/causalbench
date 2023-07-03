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

from causalscbench.third_party.dcdfg.dcdfg.utils.dag_optim import GumbelInNOut, bisect, is_acyclic


class MLPModularGaussianModule(nn.Module):
    def __init__(
        self,
        num_vars,
        num_layers,
        num_modules,
        hid_dim,
        nonlin="leaky_relu",
        constraint_mode="spectral_radius",
    ):
        """
        Simplification for the "perfect known" context and the MLP framework
        :param int num_vars: number of variables in the system
        :param int num_layers: number of hidden layers
        :param int num_modules number of modules
        :param int hid_dim: number of hidden units per layer
        :param int num_params: number of parameters per conditional *outputted by MLP*
        :param str nonlin: which nonlinearity to use
        """
        super().__init__()
        self.num_vars = num_vars
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.hid_dim = hid_dim
        self.nonlin = nonlin
        self.constraint_mode = constraint_mode

        self.weights_node2module = nn.ParameterList()
        self.weights_module2node = nn.ParameterList()
        self.biases_node2module = nn.ParameterList()
        self.biases_module2node = nn.ParameterList()
        self.log_stds = nn.Parameter(data=torch.zeros((self.num_vars,)))

        self.gumbel_innout = GumbelInNOut(self.num_vars, self.num_modules)
        self.zero_weights_ratio = 0.0
        self.numel_weights = 0

        # Instantiate the parameters of each layer in the model of each variable
        # Here, features -> modules is a MLP but modules -> gene is linear
        for weights, biases, num_out_nodes, num_in_nodes in (
            (
                self.weights_node2module,
                self.biases_node2module,
                self.num_modules,
                self.num_vars,
            ),
            # (
            #     self.weights_module2node,
            #     self.biases_module2node,
            #     self.num_vars,
            #     self.num_modules,
            # ),
        ):
            for i in range(self.num_layers + 1):
                in_dim = num_in_nodes if i == 0 else self.hid_dim
                out_dim = 1 if i == self.num_layers else self.hid_dim

                weights.append(
                    nn.Parameter(torch.zeros(num_out_nodes, out_dim, in_dim))
                )
                biases.append(nn.Parameter(torch.zeros(num_out_nodes, out_dim)))
                self.numel_weights += self.num_vars * out_dim * in_dim

            # init params
            with torch.no_grad():
                for node in range(num_out_nodes):
                    for i, w in enumerate(weights):
                        w = w[node]
                        nn.init.xavier_uniform_(
                            w, gain=nn.init.calculate_gain(self.nonlin)
                        )
                    for i, b in enumerate(biases):
                        b = b[node]
                        b.zero_()

            # separate for linear decoding model
            self.weights_module2node.append(
                nn.Parameter(torch.zeros(self.num_vars, 1, self.num_modules))
            )
            self.biases_module2node.append(nn.Parameter(torch.zeros(self.num_vars, 1)))
            with torch.no_grad():
                for node in range(num_out_nodes):
                    nn.init.xavier_uniform_(
                        self.weights_module2node[0][node],
                        gain=nn.init.calculate_gain(self.nonlin),
                    )
                    self.biases_module2node[0][node].zero_()

        # Initialization for spectral radius constraint
        w_adj = self.get_w_adj()
        self.register_buffer("u", torch.zeros(w_adj.shape[0]))
        self.register_buffer("v", torch.zeros(w_adj.shape[0]))
        a, b = -3, 3
        with torch.no_grad():
            nn.init.trunc_normal_(self.u, a=a, b=b)
            nn.init.trunc_normal_(self.v, a=a, b=b)

        # Initialization for block spectral radius constraint
        self.register_buffer("u_v", torch.zeros(self.num_vars))
        self.register_buffer("u_f", torch.zeros(self.num_modules))
        self.register_buffer("v_v", torch.zeros(self.num_vars))
        self.register_buffer("v_f", torch.zeros(self.num_modules))
        a, b = -3, 3
        with torch.no_grad():
            nn.init.trunc_normal_(self.u_v, a=a, b=b)
            nn.init.trunc_normal_(self.u_f, a=a, b=b)
            nn.init.trunc_normal_(self.v_v, a=a, b=b)
            nn.init.trunc_normal_(self.v_f, a=a, b=b)

        # get scaling factor and normalization factor for penalty
        with torch.no_grad():
            mat = w_adj
            self.base_radius = self.spectral_radius_adj(mat, n_iter=100)
            self.constraint_norm = self.compute_dag_constraint(mat).item()
            if np.isinf(self.constraint_norm):
                raise ValueError("Error: constraint normalization is infinite")

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

    def spectral_radius_block(self, A, B, n_iter=5):
        """
        A is shape n_var x n_module
        B is shape n_module x n_var
        Compute the spectral norm of U,V with a power iteration.
        :param np.ndarray w_adj: the weighted adjacency matrix (each entry in [0,1])
        """
        with torch.no_grad():
            for _ in range(n_iter):
                self.u_f = F.normalize(B @ self.u_v, dim=0)
                self.u_v = F.normalize(A @ self.u_f, dim=0)
                self.v_f = F.normalize(A.T @ self.v_v, dim=0)
                self.v_v = F.normalize(B.T @ self.v_f, dim=0)
        numerator = self.v_f.T @ B @ self.u_v + self.v_v.T @ A @ self.u_f
        denominator = self.v_f.T @ self.u_f + self.v_v.T @ self.u_v
        return numerator / denominator

    def spectral_radius_iteration(self, node2module, module2node, n_iter=5):
        """
        Compute the spectral norm of w_adj with a power iteration.
        :param np.ndarray w_adj: the weighted adjacency matrix (each entry in [0,1])
        """
        with torch.no_grad():
            for _ in range(n_iter):
                # w_adj = node2module @ module2node.T - diag
                # v update: v+ \propsto w_adj v
                diag_term = self.v * torch.sum(node2module * module2node, 1)
                self.v = F.normalize(
                    node2module @ module2node.T @ self.v - diag_term, dim=0
                )
                # u update: y+ \propsto w_adj.T u
                diag_term = self.u * torch.sum(node2module * module2node, 1)
                self.u = F.normalize(
                    module2node @ node2module.T @ self.u - diag_term, dim=0
                )
        numerator = self.v.T @ node2module @ module2node.T @ self.u
        numerator -= self.v @ (self.u * torch.sum(node2module * module2node, 1))
        return numerator / (self.v @ self.u)

    def compute_dag_constraint(self, adj):
        """
        Compute the DAG constraint on weighted adjacency matrix w_adj
        :param np.ndarray w_adj: the weighted adjacency matrix (each entry in [0,1])
        """
        if self.constraint_mode == "exp":
            return torch.trace(torch.matrix_exp(adj / self.base_radius)) - self.num_vars
        elif self.constraint_mode == "spectral_radius":
            return self.spectral_radius_adj(adj)
        elif self.constraint_mode == "exptrick":
            return (
                torch.trace(
                    torch.matrix_exp(
                        self.gumbel_innout.get_proba_modules() / self.base_radius
                    )
                )
                - self.num_modules
            )
        elif self.constraint_mode == "spectraltrick":
            return self.compute_dag_constraint_spectral(
                *self.gumbel_innout.get_proba_()
            )
        else:
            raise ValueError(
                "constraint_mode needs to be in ['exp', 'spectral_radius', 'matrix_power']."
            )

    def compute_dag_constraint_power(self, w_adj):
        """
        Compute the DAG constraint DIBS style via a matrix power.
        :param np.ndarray w_adj: the weighted adjacency matrix (each entry in [0,1])
        """
        d = w_adj.shape[0]
        return (
            torch.trace(
                torch.linalg.matrix_power(
                    torch.eye(w_adj.shape[0], device=w_adj.device)
                    + w_adj / self.base_radius,
                    d,
                )
            )
            - d
        )

    def compute_dag_constraint_spectral(self, adj_node2module, adj_module2node):
        """
        Compute the DAG constraint NO-BEARS style via the spectral norm.
        :param np.ndarray w_adj: the weighted adjacency matrix (each entry in [0,1])
        """
        return self.spectral_radius_iteration(adj_node2module, adj_module2node)

    def forward(self, x):
        """
        :param x: batch_size x num_vars
        :return: batch_size x num_vars * num_params, the parameters of each variable conditional
        """
        num_batch = x.size(0)
        num_zero_weights = 0

        # Sample masks
        mask_node2module, mask_module2node = self.gumbel_innout(num_batch)
        mask_module2node = torch.transpose(mask_module2node, 1, 2)

        for weights, biases, mask in (
            (self.weights_node2module, self.biases_node2module, mask_node2module),
            (self.weights_module2node, self.biases_module2node, mask_module2node),
        ):
            # beware, num_layers could be different in or out if linear "decoder"
            num_layers = len(weights) - 1
            for layer in range(num_layers + 1):
                # First layer, apply the mask
                if layer == 0:
                    x = (
                        torch.einsum("tij,bjt,bj->bti", weights[layer], mask, x)
                        + biases[layer]
                    )
                # 2nd layer and more
                else:
                    x = torch.einsum("tij,btj->bti", weights[layer], x) + biases[layer]

                # count number of zeros
                num_zero_weights += weights[layer].numel() - weights[
                    layer
                ].nonzero().size(0)

                # apply non-linearity
                if layer != num_layers:
                    x = (
                        F.leaky_relu(x)
                        if self.nonlin == "leaky_relu"
                        else torch.sigmoid(x)
                    )
                else:
                    # Reshape to 1 (for now, might want to increase module complexity)
                    x = x.squeeze()

        self.zero_weights_ratio = num_zero_weights / float(self.numel_weights)

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

    def losses(self, x, mask):
        """
        Compute the loss. If intervention is perfect and known, remove
        the intervened targets from the loss with a mask.
        """
        log_likelihood = torch.sum(self.log_likelihood(x) * mask, dim=0) / mask.size(0)
        # constraint related
        adj = self.get_w_adj()
        h = (
            self.compute_dag_constraint(adj)
            # / self.constraint_norm
        )
        a, b = self.gumbel_innout.get_proba_()
        reg = 0.5 * (a.sum() + b.sum()) / a.numel()
        losses = (-torch.mean(log_likelihood), h, reg)
        return losses

    def threshold(self):
        # Final thresholding of all edges until DAG is found
        with torch.no_grad():
            adj = self.gumbel_innout.get_proba_features()

            def acyc(t):
                return (
                    float(
                        is_acyclic(
                            self.gumbel_innout.get_proba_features(t).cpu().numpy()
                        )
                    )
                    - 0.5
                )

            threshold = bisect(acyc, 0, 1)
            assert acyc(threshold) > 0
            self.weight_mask = self.gumbel_innout.get_proba_features(threshold)
            print(f"threshold term-wise:{threshold}")
            print(f"numel:{(self.weight_mask > 0).sum()}")
            self.gumbel_innout.freeze_threshold(threshold)

    def check_acyclicity(self):
        adj = self.get_w_adj()
        to_keep = (adj > 0.5).type_as(adj)
        return is_acyclic(to_keep.cpu().numpy())

    def get_w_adj(self):
        return self.gumbel_innout.get_proba_features()
