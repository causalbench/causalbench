"""
Copyright (C) 2022, 2023  GlaxoSmithKline plc - Mathieu Chevalley, Yusuf Roohani;

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
import numpy as np
import pandas as pd
import scanpy as sc

def get_strong_perts(supp):
    filtered = supp[supp['Number of DEGs (anderson-darling)']>50]
    filtered = filtered[filtered['percent knockdown']<=-0.3]
    filtered = filtered[filtered['number of cells (filtered)']>25]
    strong_perts = filtered['genetic perturbation'].values
    strong_perts = [s.split('_')[1] for s in strong_perts]
    return strong_perts

def filter_cells_by_pert_effect(adata, k=10):

    subset_idxs = []
    ctrl_adata = adata[adata.obs['gene'] == 'non-targeting']

    for itr, g in enumerate(adata.obs['gene'].unique()):        
        subset = adata[adata.obs['gene'] == g]

        if g == 'non-targeting':
            subset_idxs.append(subset.obs.index.values)
            continue

        try:
            gene_loc = np.where(adata.var.gene_name == g)[0][0]
            thresh = np.percentile(ctrl_adata.X[:,gene_loc],k)

            subset_idxs.append(subset.obs.index[subset.X[:,gene_loc]<=thresh].values)
        except:
            subset_idxs.append(subset.obs.index.values)

    subset_idxs = [item for sublist in subset_idxs for item in sublist]
    filtered_adata = adata[subset_idxs,:]

    return filtered_adata


def preprocess_dataset(dataset_path: str, summary_stats: pd.DataFrame = None):
    """Preprocess the Anndata dataset and extract the necessary information

    Args:
        dataset_path (string): path to Anndata dataset
        summary_stats (pandas.DataFrame): dataframe containing summary stats to filter for strong perturbations. If None, do not filter.


    Returns:
        (numpy, list, list): expression matrix, list of gene ids (columns), list of perturbed genes (rows)
    """
    data_expr_raw = sc.read(dataset_path)
    if summary_stats is not None:
        # Filter for only strong pertubations
        strong_perts = get_strong_perts(summary_stats) + ['non-targeting']
        idx_to_keep = [v in strong_perts for v in data_expr_raw.obs['gene']]
        pert_filter_k562 = data_expr_raw[idx_to_keep]
        data_expr_raw = filter_cells_by_pert_effect(pert_filter_k562)

    # Normalize data
    sc.pp.normalize_per_cell(data_expr_raw, key_n_counts='UMI_count')
    sc.pp.log1p(data_expr_raw)  
    data_expr_raw_df = data_expr_raw.to_df()
    intervened_genes = list(data_expr_raw.obs["gene_id"])
    gene_to_interventions = dict()
    for i, intervention in enumerate(intervened_genes):
        if intervention in set(data_expr_raw_df.columns.to_list()) or intervention == "non-targeting":
            gene_to_interventions.setdefault(intervention, []).append(i)
    intervened_genes_set = set()
    for gene, cell_indices in gene_to_interventions.items():
        if len(cell_indices) > 100:
            intervened_genes_set.add(gene)
    intervened_genes = ["excluded" if gene not in intervened_genes_set else gene for gene in intervened_genes]
    data_expr_raw_perturbed_only = data_expr_raw_df[data_expr_raw_df.columns[data_expr_raw_df.columns.isin(intervened_genes_set)]]
    gene_ids = data_expr_raw_perturbed_only.columns.to_list()
    return data_expr_raw_perturbed_only.to_numpy(), gene_ids, intervened_genes
