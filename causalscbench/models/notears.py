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

import causalscbench.third_party.notears.linear
import causalscbench.third_party.notears.nonlinear
import causalscbench.third_party.notears.utils
import numpy as np
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import remove_lowly_expressed_genes


class NotearsLin(AbstractInferenceModel):
    def __init__(self, lambda1: float = 0.0) -> None:
        super().__init__()
        self.lambda1 = lambda1

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        if not training_regime == TrainingRegime.Observational:
            return []
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix, gene_names, expression_threshold=0.25
        )
        causalscbench.third_party.notears.utils.set_random_seed(seed)
        adjacency = causalscbench.third_party.notears.linear.notears_linear(
            expression_matrix, lambda1=self.lambda1, max_iter=20, loss_type="l2"
        )
        indices = np.transpose(np.nonzero(adjacency))
        edges = set()
        for (i, j) in indices:
            edges.add((gene_names[i], gene_names[j]))
        return list(edges)


class NotearsMLP(AbstractInferenceModel):
    def __init__(self, lambda1: float = 0.0) -> None:
        super().__init__()
        self.lambda1 = lambda1

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        if not training_regime == TrainingRegime.Observational:
            return []
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix, gene_names, expression_threshold=0.25
        )
        causalscbench.third_party.notears.utils.set_random_seed(seed)
        model = causalscbench.third_party.notears.nonlinear.NotearsMLP(dims=[len(gene_names), 10, 1], bias=True)
        adjacency = causalscbench.third_party.notears.nonlinear.notears_nonlinear(
            model, expression_matrix, lambda1=self.lambda1, lambda2=self.lambda1, max_iter=20
        )
        indices = np.transpose(np.nonzero(adjacency))
        edges = set()
        for (i, j) in indices:
            edges.add((gene_names[i], gene_names[j]))
        return list(edges)
