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
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import (
    partion_network, remove_lowly_expressed_genes)
from causalscbench.third_party.dcdi.dcdi.data import DataManagerFile
from causalscbench.third_party.dcdi.dcdi.models.flows import \
    DeepSigmoidalFlowModel
from causalscbench.third_party.dcdi.dcdi.models.learnables import \
    LearnableModel_NonLinGaussANM
from causalscbench.third_party.dcdi.dcdi.train import train


class DCDI(AbstractInferenceModel):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix, gene_names, expression_threshold=0.5
        )
        gene_names = np.array(gene_names)
        interventions = list(interventions)

        def process_partition(partition):
            gene_names_ = gene_names[partition]
            expression_matrix_ = expression_matrix[:, partition]
            node_dict = {g: idx for idx, g in enumerate(gene_names_)}
            gene_names_set = set(gene_names_)
            subset = []
            interventions_ = []
            for idx, iv in enumerate(interventions):
                if iv in gene_names_set or iv == "non-targeting":
                    subset.append(idx)
                    interventions_.append(iv)
            expression_matrix_ = expression_matrix_[subset, :]
            gene_to_interventions = dict()
            for i, intervention in enumerate(interventions_):
                gene_to_interventions.setdefault(intervention, []).append(i)

            mask_intervention = []
            regimes = []
            j = 0
            start = 0
            data = np.zeros_like(expression_matrix_)
            for inv, indices in gene_to_interventions.items():
                targets = [] if inv == "non-targeting" else [node_dict[inv]]
                regime = 0 if inv == "non-targeting" else j + 1
                mask_intervention.extend([targets for _ in range(len(indices))])
                regimes.extend([regime for _ in range(len(indices))])
                end = start + len(indices)
                data[start:end, :] = expression_matrix_[indices, :]
                start = end
                if inv != "non-targeting":
                    j += 1

            regimes = np.array(regimes)
            train_data = DataManagerFile(
                data,
                mask_intervention,
                regimes,
                0.8,
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
                0.8,
                train=False,
                normalize=False,
                random_seed=seed,
                intervention=True,
                intervention_knowledge="known",
            )

            if self.model == "DCDI-G":
                model = LearnableModel_NonLinGaussANM(len(gene_names_),
                                                2,
                                                15,
                                                intervention=True,
                                                intervention_type="perfect",
                                                intervention_knowledge="known",
                                                num_regimes=train_data.num_regimes)
            elif self.model == "DCDI-DSF":
                model = DeepSigmoidalFlowModel(num_vars=len(gene_names_),
                                            cond_n_layers=2,
                                            cond_hid_dim=15,
                                            cond_nonlin="leaky-relu",
                                            flow_n_layers=2,
                                            flow_hid_dim=10,
                                            intervention=True,
                                            intervention_type="perfect",
                                            intervention_knowledge="known",
                                            num_regimes=train_data.num_regimes)
            else:
                raise ValueError("Model has to be in {DCDI-G, DCDI-DSF}")

            # Default from DCDI
            opt = SimpleNamespace()
            opt.train_patience = 5
            opt.train_patience_post = 5
            opt.num_train_iter = 30000
            opt.no_w_adjs_log = True
            opt.mu_init = 1e-8
            opt.gamma_init = 0.
            opt.optimizer = "rmsprop"
            opt.lr = 1e-2
            opt.train_batch_size = 64
            opt.reg_coeff = 0.1
            opt.coeff_interv_sparsity = 0
            opt.stop_crit_win = 100
            opt.h_threshold = 1e-8
            opt.omega_gamma = 1e-4
            opt.omega_mu = 0.9
            opt.mu_mult_factor = 2
            opt.lr_reinit = 1e-2
            opt.intervention = True
            opt.intervention_type = "perfect"
            opt.intervention_knowledge = "known"
            opt.gpu = True

            train(model, train_data, test_data, opt)

            adjacency = model.get_w_adj()
            indices = np.nonzero(adjacency > 0.75)
            edges = set()
            for (i, j) in indices:
                edges.add((gene_names_[i], gene_names_[j]))
            return list(edges)

        partitions = partion_network(gene_names, 50, seed)
        edges = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            partition_results = list(executor.map(process_partition, partitions))
            for result in partition_results:
                edges += result
        return edges