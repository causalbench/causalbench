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

import random
from re import A
from typing import List, Tuple

import numpy as np
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime


class RandomWithSize(AbstractInferenceModel):
    def __init__(self, size) -> None:
        super().__init__()
        self.size = size

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        random.seed(seed)
        edges = set()
        while len(edges) < self.size:
            pair = random.sample(range(len(gene_names)), 2)
            edges.add((gene_names[pair[0]], gene_names[pair[1]]))
        return list(edges)


class FullyConnected(AbstractInferenceModel):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        random.seed(seed)
        edges = set()
        for i in range(len(gene_names)):
            a = gene_names[i]
            for j in range(i + 1, len(gene_names)):
                b = gene_names[j]
                edges.add((a, b))
                edges.add((b, a))
        return list(edges)
