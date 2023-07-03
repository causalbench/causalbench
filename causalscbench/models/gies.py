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
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import gies
import numpy as np
import pandas as pd
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import (
    causallearn_graph_to_edges, partion_network, remove_lowly_expressed_genes)


class GIES(AbstractInferenceModel):
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
            data = []
            I = []
            for inv, indices in gene_to_interventions.items():
                if inv == "non-targeting":
                    I.append([])
                else:
                    I.append([node_dict[inv]])
                data.append(expression_matrix_[indices, :])

            adjacency, _ = gies.fit_bic(data, I, iterate=False)
            indices = np.transpose(np.nonzero(adjacency))
            edges = set()
            for (i, j) in indices:
                edges.add((gene_names_[i], gene_names_[j]))
            return list(edges)
            
        
        partitions = partion_network(gene_names, 30, seed)
        edges = []
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            partition_results = list(executor.map(process_partition, partitions))
            for result in partition_results:
                edges += result
        return edges