"""
Copyright (C) 2022  GlaxoSmithKline plc - Yusuf Roohani;

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

import networkx as nx
import numpy as np
from causaldag import (MemoizedCI_Tester, MemoizedInvarianceTester,
                       gauss_invariance_suffstat, gauss_invariance_test, gsp,
                       igsp, partial_correlation_suffstat,
                       partial_correlation_test, rand)
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime


class GreedySparsestPermutation(AbstractInferenceModel):
    """Network inference model based on GSP."""

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
        if not training_regime == TrainingRegime.Observational:
            return []

        edges = set()
        nodes = list(range(len(gene_names)))
        suffstat = partial_correlation_suffstat(expression_matrix)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3)
        dag = gsp(nodes, ci_tester)
        
        ## Convert edges to correct format
        for edge in nx.generate_adjlist(dag.to_nx()):
            edge_nodes = [int(e) for e in edge.split(' ')]
            if len(edge_nodes)>1:
                edges.add((gene_names[edge_nodes[0]], gene_names[edge_nodes[1]]))
    
        return list(edges)


class InterventionalGreedySparsestPermutation(AbstractInferenceModel):
    """Network inference model based on GSP."""

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
        
        edges = set()
        node_dict = {g:idx for idx, g in enumerate(gene_names)}
        nodes = list(range(len(gene_names)))
        
        # Create list of interventional samples
        interventions = [i for i in interventions if (i in gene_names) and (i != "non-targeting")]
        iv_samples_list = [expression_matrix[np.where(np.array(interventions)==i)[0], :] 
                           for i in interventions]
        obs_samples = expression_matrix[np.where(np.array(interventions)=="non-targeting")[0], :]
        setting_list = [{'interventions': [node_dict[i]]} for i in interventions]
        
        # Sufficient statistics for observational and interventional data
        obs_suffstat = partial_correlation_suffstat(expression_matrix)
        inv_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)
        
        # CI tester and invariance tester
        ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=1e-3)
        inv_tester = MemoizedInvarianceTester(gauss_invariance_test, inv_suffstat, 
                                              alpha=1e-3)
        
        # Estimate DAG
        dag = igsp(
            setting_list,
            nodes,
            ci_tester,
            inv_tester
        )
        
        ## Convert edges to correct format
        for edge in nx.generate_adjlist(dag.to_nx()):
            edge_nodes = [int(e) for e in edge.split(' ')]
            if len(edge_nodes)>1:
                edges.add((gene_names[edge_nodes[0]], gene_names[edge_nodes[1]]))
    
        return list(edges)

