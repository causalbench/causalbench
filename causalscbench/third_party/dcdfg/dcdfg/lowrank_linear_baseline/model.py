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
import pytorch_lightning as pl
import torch

from causalscbench.third_party.dcdfg.dcdfg.lowrank_linear_baseline.module import LinearModularGaussianModule


class LinearModuleGaussianModel(pl.LightningModule):
    """
    Lightning module that runs augmented lagrangian
    """

    def __init__(
        self,
        num_vars,
        num_modules,
        lr_init=1e-3,
        reg_coeff=0.1,
        constraint_mode="exp",
    ):
        super().__init__()
        self.module = LinearModularGaussianModule(
            num_vars,
            num_modules,
            constraint_mode=constraint_mode,
        )
        # augmented lagrangian params
        # mu: penalty
        # gamma: multiplier
        self.mu_init = 1e-8
        self.gamma_init = 0.0
        self.omega_gamma = 1e-4
        self.omega_mu = 0.9
        self.h_threshold = 1e-8
        self.mu_mult_factor = 2

        # opt params
        self.save_hyperparameters()
        self.lr_init = lr_init
        self.reg_coeff = reg_coeff
        self.constraint_mode = constraint_mode
        self.hparams["name"] = self.__class__.__name__
        self.hparams["module_name"] = self.module.__class__.__name__

        # initialize stuff for learning loop
        self.aug_lagrangians = []
        self.not_nlls = []  # Augmented Lagrangrian minus (pseudo) NLL
        self.nlls = []  # NLL on train
        self.nlls_val = []  # NLL on validation
        self.regs = []

        # Augmented Lagrangian stuff
        self.mu = self.mu_init
        self.gamma = self.gamma_init

        # bookkeeping for training
        self.acyclic = 0.0
        self.aug_lagrangians_val = []
        self.not_nlls_val = []
        self.constraint_value = 0.0
        self.constraints_at_stat = []
        self.reg_value = 0.0
        self.internal_checkups = 0.0
        self.stationary_points = 0.0

        self.validation_step_outputs = []

    def forward(self, data):
        x, masks, regimes = data
        log_likelihood = torch.sum(
            self.module.log_likelihood(x) * masks, dim=0
        ) / masks.size(0)
        return -torch.mean(log_likelihood)

    def get_augmented_lagrangian(self, nll, constraint_violation, reg):
        # compute augmented langrangian
        return (
            nll
            + self.reg_coeff * reg
            + self.gamma * constraint_violation
            + 0.5 * self.mu * constraint_violation**2
        )

    def training_step(self, batch, batch_idx):
        # get data
        x, masks, regimes = batch

        # compute loss
        nll, constraint_violation, reg = self.module.losses(x, masks)
        aug_lagrangian = self.get_augmented_lagrangian(nll, constraint_violation, reg)

        # logging
        self.nlls.append(nll.item())
        self.aug_lagrangians.append(aug_lagrangian.item())
        self.not_nlls.append(aug_lagrangian.item() - nll.item())

        self.log("Train/aug_lagrangian", aug_lagrangian.detach())
        self.log("Train/nll", nll.detach())
        self.log("Train/not_nll", aug_lagrangian.detach() - nll.detach())
        self.log("Aug_lag/mu", self.mu)
        self.log("Aug_lag/gamma", self.gamma)

        # return loss
        return aug_lagrangian

    def validation_step(self, batch, batch_idx):
        x, masks, regimes = batch
        nll, constraint_violation, reg = self.module.losses(x, masks)
        aug_lagrangian = self.get_augmented_lagrangian(nll, constraint_violation, reg)
        outputs = {
            "aug_lagrangian": aug_lagrangian,
            "nll": nll,
            "constraint": constraint_violation,
            "reg": reg,
        }
        self.validation_step_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self):
        agg = {}
        for k in self.validation_step_outputs[0]:
            agg[k] = torch.stack([dic[k] for dic in self.validation_step_outputs]).mean().item()
        self.aug_lagrangians_val += [agg["aug_lagrangian"]]
        self.constraint_value = agg["constraint"]
        self.reg_value = agg["reg"]
        self.not_nlls_val += [agg["aug_lagrangian"] - agg["nll"]]
        self.nlls_val += [agg["nll"]]
        self.regs += [self.reg_value]
        # self.acyclic = self.module.check_acyclicity()

        self.log("Val/aug_lagrangian", agg["aug_lagrangian"])
        self.log("Val/nll", agg["nll"])
        self.log("Val/not_nll", agg["aug_lagrangian"] - agg["nll"])
        # self.log("Val/acyclic", float(self.acyclic))
        self.log("Val/constraint_violation", agg["constraint"])
        self.log("Val/reg_value", agg["reg"])
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.RMSprop(self.module.parameters(), lr=self.lr_init)

    def update_lagrangians(self):
        self.internal_checkups += 1
        self.log("Monitor/checkup", self.internal_checkups)
        # compute delta for gamma to check convergence status
        delta_gamma = -np.inf
        if len(self.aug_lagrangians_val) >= 3:
            t0, t_half, t1 = (
                self.aug_lagrangians_val[-3],
                self.aug_lagrangians_val[-2],
                self.aug_lagrangians_val[-1],
            )
            # if the validation loss went up and down, do not update lagrangian and penalty coefficients.
            if min(t0, t1) < t_half < max(t0, t1):
                delta_gamma = -np.inf
            else:
                delta_gamma = (t1 - t0) / 100

        # if we found a stationary point, but that is not satisfying the acyclicity constraints
        if (
            self.constraint_value > self.h_threshold
            and not self.acyclic
            and self.mu < 1e15
            or self.stationary_points < 10
        ):
            if abs(delta_gamma) < self.omega_gamma or delta_gamma > 0:
                self.stationary_points += 1
                self.log("Monitor/stationary", self.stationary_points)
                self.gamma += self.mu * self.constraint_value

                # Did the constraint improve sufficiently?
                if len(self.constraints_at_stat) > 1:
                    if (
                        self.constraint_value
                        > self.constraints_at_stat[-1] * self.omega_mu
                    ):
                        self.mu *= self.mu_mult_factor
                self.constraints_at_stat.append(self.constraint_value)

                # little hack to make sure the moving average is going down.
                gap_in_not_nll = (
                    self.get_augmented_lagrangian(
                        0.0, self.constraint_value, self.reg_value
                    )
                    - self.not_nlls_val[-1]
                )
                assert gap_in_not_nll > -1e-2
                self.aug_lagrangians_val[-1] += gap_in_not_nll

                # reset optimizer
                self.trainer.optimizers = [self.configure_optimizers()]

        # if we found a stationary point, that satisfies the acyclicity constraints, raise this flag, it will activate patience and terminate training soon
        else:
            self.trainer.satisfied_constraints = True
