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
from typing import List

import numpy as np
import pandas as pd
import scprep
from causallearn.graph import GeneralGraph


def causallearn_graph_to_edges(G: GeneralGraph, gene_names: List[str]):
    node_map = G.get_node_map()
    edges = []
    for edge in G.get_graph_edges():
        node1 = edge.get_node1()
        node2 = edge.get_node2()
        if gene_names is None:
            edges.append((node1.get_name(), node2.get_name()))
        else:
            node1_id = node_map[node1]
            node2_id = node_map[node2]
            edges.append((gene_names[node1_id], gene_names[node2_id]))
    return edges

def partion_network(gene_names, partitions_length, seed):
    random.seed(seed)
    indices = list(range(len(gene_names)))
    random.shuffle(indices)
    partition_length = int(len(indices) / partitions_length)
    partitions = [indices[i::partition_length] for i in range(partition_length)]
    return partitions

def remove_lowly_expressed_genes(expression_matrix: np.array, gene_names: List[str], expression_threshold=0.8):
    """Remove genes with low expression.

    Args:
        expression_matrix (np.array): Expression matrix with cells as index and 
        genes_names: name of genes corresponding to the columns of the expression matrix
        expression_threshold (float, optional): Min level of expression across cells in percent. Defaults to 0.8.

    Returns:
        DataFrame: Expression matrix with only highly expressed genes
    """
    min_cells = expression_matrix.shape[0] * expression_threshold
    expression_matrix = pd.DataFrame(expression_matrix, columns=gene_names)
    subsample_genes = scprep.filter.filter_rare_genes(
        expression_matrix, min_cells=int(min_cells))
    return subsample_genes.to_numpy(), subsample_genes.columns.to_list()
