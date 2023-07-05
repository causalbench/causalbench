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

from typing import List, Tuple

import distributed
import numpy as np
from arboreto import algo
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import remove_lowly_expressed_genes


class GRNBoost(AbstractInferenceModel):
    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix, gene_names, expression_threshold=0.25
        )

        local_cluster = distributed.LocalCluster(n_workers=25, threads_per_worker=5)

        custom_client = distributed.Client(local_cluster)
        network = algo.grnboost2(
            expression_data=expression_matrix,
            gene_names=gene_names,
            client_or_address=custom_client,
            seed=seed,
            early_stop_window_length=10,
            verbose=True,
        )
        return [(i, j) for i, j in network[["TF", "target"]].values]


class GENIE(AbstractInferenceModel):
    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        local_cluster = distributed.LocalCluster(n_workers=15, threads_per_worker=10)
        custom_client = distributed.Client(local_cluster)
        network = algo.genie3(
            expression_data=expression_matrix,
            gene_names=gene_names,
            client_or_address=custom_client,
            seed=seed,
            verbose=True,
        )
        return [(i, j) for i, j in network[["TF", "target"]].values]
