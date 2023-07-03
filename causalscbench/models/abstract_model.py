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

from abc import abstractmethod
from typing import List, Tuple

import numpy as np
from causalscbench.models.training_regimes import TrainingRegime


class AbstractInferenceModel(object):
    @abstractmethod
    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0
    ) -> List[Tuple]:
        """
        Learn a GRN causal network given single cell expression data.

        Args:
            expression_matrix: a numpy matrix of expression data of size [nb_samples, nb_genes]
            interventions: a list of size [nb_samples] that indicates which gene has been perturb. "non-targeting" means no gene has been perturbed (observational data)
            gene_names: name of the genes in the expression matrix
            training_regime: indicates in which training regime we are (either fully observational, partially intervened or fully intervened)
            seed: randomness seed for reproducibility
        Returns:
            A list of pairs of gene (from gene_names). (A, B) means that there is an edge from A to B in the inferred network

        """
        raise NotImplementedError()
