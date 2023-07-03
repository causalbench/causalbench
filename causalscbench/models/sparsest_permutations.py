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

import pdb
import networkx as nx
import numpy as np
from causaldag import (MemoizedCI_Tester, MemoizedInvarianceTester,
                       gauss_invariance_suffstat, gauss_invariance_test, gsp,
                       igsp, partial_correlation_suffstat, rand)

from causalscbench.third_party.causaldag import partial_correlation_test

from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from causalscbench.models.utils.model_utils import remove_lowly_expressed_genes

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
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix, gene_names, expression_threshold=0.25
        )
        edges = set()
        nodes = list(range(len(gene_names)))
        suffstat = partial_correlation_suffstat(expression_matrix)
        ci_tester = MemoizedCI_Tester(partial_correlation_test, suffstat, alpha=1e-3, 
                                      track_times=True)
        dag = gsp(set(nodes), ci_tester, depth=2)
        
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
        expression_matrix, gene_names = remove_lowly_expressed_genes(
            expression_matrix, gene_names, expression_threshold=0.5
        )
        node_dict = {g:idx for idx, g in enumerate(gene_names)}
        nodes = list(range(len(gene_names)))

        # Observational samples
        interventions = [i for i in interventions]
        obs_idxs = np.where(np.array(interventions)=="non-targeting")[0]
        obs_samples = expression_matrix[obs_idxs, :]

        # Create list of interventional samples
        interventions = [i for i in interventions if (i in gene_names) and (i != "non-targeting")]
        interventions_unique = list(set(interventions).difference(set(['non-targeting'])))
        iv_idxs = [np.where(np.array(interventions)==i)[0] for i in interventions_unique]
        
        # Create list of interventional samples and remove lists with only a single sample
        iv_samples_list = []
        intv_to_remove = []
        for iv_idx, intv_name in zip(iv_idxs, interventions_unique):
            if len(iv_idx)>1:
                iv_samples_list.append(expression_matrix[iv_idx, :])
            else:
                intv_to_remove.append(intv_name)
        interventions = [x for x in interventions if x not in intv_to_remove]
        interventions_unique = list(set(interventions_unique).difference(set(intv_to_remove)))  

        setting_list = [{'interventions': [node_dict[i]]} for i in interventions_unique]
        
        # Sufficient statistics for observational and interventional data
        obs_suffstat = partial_correlation_suffstat(obs_samples)
        inv_suffstat = gauss_invariance_suffstat(obs_samples, iv_samples_list)
        
        # CI tester and invariance tester
        ci_tester = MemoizedCI_Tester(partial_correlation_test, obs_suffstat, alpha=1e-3)
        inv_tester = MemoizedInvarianceTester(gauss_invariance_test, inv_suffstat, 
                                              alpha=1e-3)
       
        # Estimate DAG
        dag = igsp(
            setting_list,
            set(nodes),
            ci_tester,
            inv_tester
        )
        
        ## Convert edges to correct format
        for edge in nx.generate_adjlist(dag.to_nx()):
            edge_nodes = [int(e) for e in edge.split(' ')]
            if len(edge_nodes)>1:
                edges.add((gene_names[edge_nodes[0]], gene_names[edge_nodes[1]]))
    
        return list(edges)
