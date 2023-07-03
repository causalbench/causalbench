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
from numba import njit
from tqdm import tqdm

from causalscbench.third_party.dcdfg.dcdfg.utils.gumbel import gumbel_sigmoid, gumbel_softmax


@njit
def samesign(a, b):
    return a * b > 0


def bisect(func, low, high, T=20):
    "Find root of continuous function where f(low) and f(high) have opposite signs"
    flow = func(low)
    fhigh = func(high)
    assert not samesign(flow, fhigh)
    for i in tqdm(range(T), desc="bisecting"):
        midpoint = (low + high) / 2.0
        fmid = func(midpoint)
        if samesign(flow, fmid):
            low = midpoint
            flow = fmid
        else:
            high = midpoint
            fhigh = fmid
    # after all those iterations, low has one sign, and high another one. midpoint is unknown
    return high


@njit
def _is_acyclic(adjacency):
    """
    Return true if adjacency is a acyclic
    :param np.ndarray adjacency: adjacency matrix
    """
    prod = np.eye(adjacency.shape[0], dtype=adjacency.dtype)
    for _ in range(1, adjacency.shape[0] + 1):
        prod = adjacency @ prod
        if np.trace(prod) != 0:
            return False
    return True


def is_acyclic(adjacency):
    return _is_acyclic(adjacency.astype(float))


class GumbelAdjacency(torch.nn.Module):
    """
    Probabilistic mask used for DAG learning.
    Can sample a matrix and backpropagate using the
    Gumbel straigth-through estimator.
    :param int num_vars: number of variables
    """

    def __init__(self, num_rows, num_cols=None):
        super(GumbelAdjacency, self).__init__()
        if num_cols is None:
            # square matrix
            self.num_vars = (num_rows, num_rows)
        else:
            self.num_vars = (num_rows, num_cols)
        self.log_alpha = torch.nn.Parameter(torch.zeros(self.num_vars))
        self.tau = 1
        self.reset_parameters()

    def forward(self, bs):
        adj = gumbel_sigmoid(self.log_alpha, bs, tau=self.tau, hard=True)
        return adj

    def get_proba(self):
        """Returns probability of getting one"""
        return torch.sigmoid(self.log_alpha / self.tau)

    def reset_parameters(self):
        torch.nn.init.constant_(self.log_alpha, 5)


class GumbelInNOut(torch.nn.Module):
    """
    Random matrix M used for encoding egdes between modules and genes.
    Category:
    - 0 means no edge
    - 1 means node2module edge
    - 2 means module2node edge
    Can sample a matrix and backpropagate using the
    Gumbel straigth-through estimator.
    :param int num_vars: number of variables
    """

    def __init__(self, num_nodes, num_modules):
        super(GumbelInNOut, self).__init__()
        self.num_vars = (num_nodes, num_modules)
        self.log_alpha = torch.nn.Parameter(torch.zeros(num_nodes, num_modules, 3))
        self.register_buffer(
            "freeze_node2module",
            torch.zeros((num_nodes, num_modules)),
        )
        self.register_buffer(
            "freeze_module2node",
            torch.zeros((num_nodes, num_modules)),
        )
        self.tau = 1
        self.drawhard = True
        self.deterministic = False
        self.reset_parameters()

    def forward(self, bs):
        if not self.deterministic:
            design = gumbel_softmax(
                self.log_alpha, bs, tau=self.tau, hard=self.drawhard
            )
            node2module = design[:, :, :, 0]
            module2node = design[:, :, :, 1]
        else:
            node2module = self.freeze_node2module.unsqueeze(0)
            module2node = self.freeze_module2node.unsqueeze(0)
        return node2module, module2node

    def freeze_threshold(self, threshold):
        """Returns probability of being assigned into a bucket"""
        design = torch.softmax(self.log_alpha / self.tau, -1)
        node2module = design[:, :, 0]
        module2node = design[:, :, 1]
        max_in_out = torch.maximum(node2module, module2node)
        # zero for low confidence
        mask_keep = max_in_out >= threshold
        # track argmax
        self.freeze_node2module = (node2module == max_in_out) * mask_keep
        self.freeze_module2node = (module2node == max_in_out) * mask_keep
        self.deterministic = True
        print("Freeze threshold:" + str(self.freeze_module2node.device))

    def get_proba_modules(self):
        """Returns probability of being assigned into a bucket"""
        design = torch.softmax(self.log_alpha / self.tau, -1)
        node2module = design[:, :, 0]
        module2node = design[:, :, 1]
        mat = module2node.T @ node2module
        # above is correct except for diagonal values (individual values in the matrix product are corr.)
        mask_modules = torch.ones(self.num_vars[1], self.num_vars[1]) - torch.eye(
            self.num_vars[1]
        )
        return mat * mask_modules.type_as(mat)

    def get_proba_features(self, threshold=None):
        """Returns probability of being assigned into a bucket"""
        design = torch.softmax(self.log_alpha / self.tau, -1)
        node2module = design[:, :, 0]
        module2node = design[:, :, 1]
        if not threshold:
            # return a differentiable tensor
            mat = node2module @ module2node.T
            # above is correct except for diagonal values (individual values in the matrix product are corr.)
            mask_nodes = torch.ones(self.num_vars[0], self.num_vars[0]) - torch.eye(
                self.num_vars[0]
            )
            return mat * mask_nodes.type_as(mat)
        else:
            # here return a matrix without grad
            # we're thresholding here according to the edge direction confidence
            max_in_out = torch.maximum(design[:, :, 0], design[:, :, 1])
            # zero for low confidence
            mask_keep = design[:, :, 0] + design[:, :, 1] >= threshold
            # track argmax
            node2module = (design[:, :, 0] == max_in_out) * mask_keep
            module2node = (design[:, :, 1] == max_in_out) * mask_keep
            # that product below has no self cycles
            return (
                node2module.type_as(self.log_alpha)
                @ module2node.type_as(self.log_alpha).T
            )

    def get_proba_(self):
        design = torch.softmax(self.log_alpha / self.tau, -1)
        node2module = design[:, :, 0]
        module2node = design[:, :, 1]
        return node2module, module2node

    def reset_parameters(self):
        torch.nn.init.constant_(self.log_alpha, 1)
