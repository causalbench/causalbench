"""
Copyright (C) 2022  GlaxoSmithKline plc - Mathieu Chevalley;

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from torch.utils.data import DataLoader, random_split
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import (
    partion_network,
    remove_lowly_expressed_genes,
)
from causalscbench.third_party.dcdi.dcdi.data import DataManagerFile
from causalscbench.third_party.dcdi.dcdi.models.flows import DeepSigmoidalFlowModel
from causalscbench.third_party.dcdi.dcdi.models.learnables import (
    LearnableModel_NonLinGaussANM,
)
from causalscbench.third_party.dcdi.dcdi.train import train

from causalscbench.third_party.dcdfg.dcdfg.callback import (
    AugLagrangianCallback,
    ConditionalEarlyStopping,
)
from causalscbench.third_party.dcdfg.dcdfg.linear_baseline.model import (
    LinearGaussianModel,
)
from causalscbench.third_party.dcdfg.dcdfg.lowrank_linear_baseline.model import (
    LinearModuleGaussianModel,
)
from causalscbench.third_party.dcdfg.dcdfg.lowrank_mlp.model import (
    MLPModuleGaussianModel,
)


class DCDI(AbstractInferenceModel):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

        self.opt = SimpleNamespace()
        self.opt.train_patience = 5
        self.opt.train_patience_post = 5
        self.opt.num_train_iter = 60000
        self.opt.no_w_adjs_log = True
        self.opt.mu_init = 1e-8
        self.opt.gamma_init = 0.0
        self.opt.optimizer = "rmsprop"
        self.opt.lr = 1e-2
        self.opt.train_batch_size = 64
        self.opt.reg_coeff = 0.1
        self.opt.coeff_interv_sparsity = 0
        self.opt.stop_crit_win = 100
        self.opt.h_threshold = 1e-8
        self.opt.omega_gamma = 1e-4
        self.opt.omega_mu = 0.9
        self.opt.mu_mult_factor = 2
        self.opt.lr_reinit = 1e-2
        self.opt.intervention = True
        self.opt.intervention_type = "perfect"
        self.opt.intervention_knowledge = "known"
        self.opt.gpu = True

        self.gene_expression_threshold = 0.5
        self.soft_adjacency_matrix_threshold = 0.5

        self.fraction_train_data = 0.8

        self.gene_partition_sizes = 50
        self.max_parallel_executors = 16

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        """
            expression_matrix: numpy array of size n_samples x n_genes, which contains the expression values
                                of each gene in different cells
            interventions: list of size n_samples. Indicates which gene has been perturbed in each sample.
                            If value is "non-targeting", no gene was targeted (observational sample).
                            If value is "excluded", a gene was perturbed which is not in gene_names (a confounder was perturbed).
                            You may want to exclude those samples or still try to leverage them.
            gene_names: names of the genes of size n_genes. To be used as node names for the output graph.


        Returns:
            List of string tuples: output graph as list of edges.
        """
        # We remove genes that have a non-zero expression in less than 25% of samples.
        # You may want to select the genes differently.
        # You could also preprocess the expression matrix, for example to impute 0.0 expression values.
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix,
            gene_names,
            expression_threshold=self.gene_expression_threshold,
        )
        gene_names = np.array(gene_names)

        if self.opt.gpu:
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

        def process_partition(partition):
            gene_names_ = gene_names[partition]
            expression_matrix_ = expression_matrix[:, partition]
            node_dict = {g: idx for idx, g in enumerate(gene_names_)}
            gene_names_set = set(gene_names_)
            subset = []
            interventions_ = []
            for idx, intervention in enumerate(interventions):
                if intervention in gene_names_set or intervention == "non-targeting":
                    subset.append(idx)
                    interventions_.append(intervention)
            expression_matrix_ = expression_matrix_[subset, :]
            gene_to_interventions = dict()
            for i, intervention in enumerate(interventions_):
                gene_to_interventions.setdefault(intervention, []).append(i)

            mask_intervention = []
            regimes = []
            regime_index = 0
            start = 0
            data = np.zeros_like(expression_matrix_)
            for inv, indices in gene_to_interventions.items():
                targets = [] if inv == "non-targeting" else [node_dict[inv]]
                regime = 0 if inv == "non-targeting" else regime_index + 1
                mask_intervention.extend([targets for _ in range(len(indices))])
                regimes.extend([regime for _ in range(len(indices))])
                end = start + len(indices)
                data[start:end, :] = expression_matrix_[indices, :]
                start = end
                if inv != "non-targeting":
                    regime_index += 1

            regimes = np.array(regimes)

            train_data = DataManagerFile(
                data,
                mask_intervention,
                regimes,
                self.fraction_train_data,
                train=True,
                normalize=False,
                random_seed=seed,
                intervention=True,
                intervention_knowledge="known",
            )
            test_data = DataManagerFile(
                data,
                mask_intervention,
                regimes,
                self.fraction_train_data,
                train=False,
                normalize=False,
                random_seed=seed,
                intervention=True,
                intervention_knowledge="known",
            )

            # You may want to play around with the hyper parameters to find the optimal ones.
            if self.model == "DCDI-G":
                model = LearnableModel_NonLinGaussANM(
                    num_vars=len(gene_names_),
                    num_layers=2,
                    hid_dim=15,
                    intervention=True,
                    intervention_type=self.opt.intervention_type,
                    intervention_knowledge=self.opt.intervention_knowledge,
                    num_regimes=train_data.num_regimes,
                )
            elif self.model == "DCDI-DSF":
                model = DeepSigmoidalFlowModel(
                    num_vars=len(gene_names_),
                    cond_n_layers=2,
                    cond_hid_dim=15,
                    cond_nonlin="leaky-relu",
                    flow_n_layers=2,
                    flow_hid_dim=10,
                    intervention=True,
                    intervention_type=self.opt.intervention_type,
                    intervention_knowledge=self.opt.intervention_knowledge,
                    num_regimes=train_data.num_regimes,
                )
            else:
                raise ValueError("Model has to be in {DCDI-G, DCDI-DSF}")

            train(model, train_data, test_data, self.opt)

            adjacency = model.get_w_adj()
            indices = np.nonzero(adjacency > self.soft_adjacency_matrix_threshold)
            edges = set()
            for (i, j) in indices:
                edges.add((gene_names_[i], gene_names_[j]))
            return list(edges)

        partitions = partion_network(gene_names, self.gene_partition_sizes, seed)
        edges = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_executors) as executor:
            partition_results = list(executor.map(process_partition, partitions))
            for result in partition_results:
                edges += result
        return edges


class DCDFG(AbstractInferenceModel):
    def __init__(self, model_name) -> None:
        super().__init__()

        self.opt = SimpleNamespace()
        self.opt.trainsamples = 0.8
        self.opt.train_batch_size = 64
        self.opt.num_train_epochs = 600
        self.opt.num_fine_epochs = 50
        self.opt.num_modules = 10
        self.opt.lr = 1e-3
        self.opt.reg_coeff = 0.1 #0.1
        self.opt.constraint_mode = "exp"
        self.opt.model = model_name
        self.opt.poly = False
        self.gene_expression_threshold = 0.25

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix,
            gene_names,
            expression_threshold=self.gene_expression_threshold,
        )

        train_dataset = WeissmannDataset(expression_matrix, gene_names, interventions)

        nb_nodes = len(gene_names)

        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        if self.opt.model == "linear":
            model = LinearGaussianModel(
                nb_nodes,
                lr_init=self.opt.lr,
                reg_coeff=self.opt.reg_coeff,
                constraint_mode=self.opt.constraint_mode,
                poly=self.opt.poly,
            )
        elif self.opt.model == "linearlr":
            model = LinearModuleGaussianModel(
                nb_nodes,
                self.opt.num_modules,
                lr_init=self.opt.lr,
                reg_coeff=self.opt.reg_coeff,
                constraint_mode=self.opt.constraint_mode,
            )
        elif self.opt.model == "mlplr":
            model = MLPModuleGaussianModel(
                nb_nodes,
                num_layers=2,
                num_modules=self.opt.num_modules,
                hid_dim=16,
                lr_init=self.opt.lr,
                reg_coeff=self.opt.reg_coeff,
                constraint_mode=self.opt.constraint_mode,
            )
        else:
            raise ValueError("couldn't find model")

        early_stop_1_callback = ConditionalEarlyStopping(
            monitor="Val/aug_lagrangian",
            min_delta=1e-4,
            patience=5,
            verbose=True,
            mode="min",
        )
        trainer = pl.Trainer(
            max_epochs=self.opt.num_train_epochs,
            logger=None,
            val_check_interval=1.0,
            callbacks=[AugLagrangianCallback(), early_stop_1_callback],
        )
        trainer.fit(
            model,
            DataLoader(
                train_dataset, batch_size=self.opt.train_batch_size, num_workers=4, drop_last=True
            ),
            DataLoader(val_dataset, num_workers=8, batch_size=256, drop_last=True),
        )

        model.module.threshold()
        model.module.constraint_mode = "exp"
        model.gamma = 0.0
        model.mu = 0.0

        early_stop_2_callback = EarlyStopping(
            monitor="Val/nll", min_delta=1e-6, patience=5, verbose=False, mode="min"
        )
        trainer_fine = pl.Trainer(
            max_epochs=self.opt.num_fine_epochs,
            logger=None,
            val_check_interval=1.0,
            callbacks=[early_stop_2_callback],
        )
        trainer_fine.fit(
            model,
            DataLoader(train_dataset, batch_size=self.opt.train_batch_size, drop_last=True),
            DataLoader(val_dataset, num_workers=2, batch_size=256, drop_last=True),
        )

        pred_adj = model.module.weight_mask.detach().cpu().numpy()

        indices = np.nonzero(pred_adj)
        edges = set()
        for i, j in zip(indices[0], indices[1]):
            edges.add((gene_names[i], gene_names[j]))
        return list(edges)


class WeissmannDataset(Dataset):
    """
    A generic class for simulation data loading and extraction
    NOTE: the 0-th regime should always be the observational one
    """

    def __init__(
        self,
        expression_matrix,
        var_names,
        perturbations,
    ) -> None:
        """
        :param numpy.ndarray data: Expression matrix of shape (num_samples, num_genes)
        :param list var_names: List of variable names corresponding to columns in the data matrix
        :param list perturbations: List of lists indicating which variables are perturbed for each sample
        """
        super(WeissmannDataset, self).__init__()
        self.var_names = var_names

        node_dict = {g: idx for idx, g in enumerate(self.var_names)}

        gene_to_interventions = dict()
        gene_names_set = set(var_names)
        for i, intervention in enumerate(perturbations):
            if intervention in gene_names_set or intervention == "non-targeting":
                gene_to_interventions.setdefault(intervention, []).append(i)

        masks = []
        regimes = []
        regime_index = 0
        start = 0
        data = np.zeros_like(expression_matrix)
        for inv, indices in gene_to_interventions.items():
            targets = [] if inv == "non-targeting" else [node_dict[inv]]
            regime = 0 if inv == "non-targeting" else regime_index + 1
            masks.extend([targets for _ in range(len(indices))])
            regimes.extend([regime for _ in range(len(indices))])
            end = start + len(indices)
            data[start:end, :] = expression_matrix[indices, :]
            start = end
            if inv != "non-targeting":
                regime_index += 1

        self.regimes = regimes
        self.masks = np.array(masks, dtype=object)
        self.data = data

        self.num_regimes = np.unique(self.regimes).shape[0]
        self.num_samples = self.masks.shape[0]
        self.dim = self.data.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        masks_list = self.masks[idx]
        masks = np.ones((self.dim,))
        for j in masks_list:
            masks[j] = 0
        return (
            self.data[idx],
            masks,
            self.regimes[idx],
        )
