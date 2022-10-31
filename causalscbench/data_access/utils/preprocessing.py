"""
Copyright 2021 GSK plc

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
import scanpy as sc


def preprocess_dataset(dataset_path):
    """Preprocess the Anndata dataset and extract the necessary information

    Args:
        dataset_path (string): path to Anndata dataset

    Returns:
        (numpy, list, list): expression matrix, list of gene ids (columns), list of perturbed genes (rows)
    """
    data_expr_raw = sc.read(dataset_path)
    # Normalize data
    sc.pp.normalize_per_cell(data_expr_raw, key_n_counts='UMI_count')
    sc.pp.log1p(data_expr_raw)  
    intervened_genes = list(data_expr_raw.obs["gene_id"])
    gene_to_interventions = dict()
    for i, intervention in enumerate(intervened_genes):
        gene_to_interventions.setdefault(intervention, []).append(i)
    intervened_genes_set = set()
    for gene, cell_indices in gene_to_interventions.items():
        if len(cell_indices) > 100:
            intervened_genes_set.add(gene)
    for i in range(len(intervened_genes)):
        if intervened_genes[i] not in intervened_genes_set:
            intervened_genes[i] = "excluded"
    data_expr_raw_df = data_expr_raw.to_df()
    data_expr_raw_perturbed_only = data_expr_raw_df[data_expr_raw_df.columns[data_expr_raw_df.columns.isin(intervened_genes_set)]]
    gene_ids = data_expr_raw_perturbed_only.columns.to_list()
    return data_expr_raw_perturbed_only.to_numpy(), gene_ids, intervened_genes
