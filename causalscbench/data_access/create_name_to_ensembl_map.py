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
from typing import Dict

import scanpy as sc
from causalscbench.data_access.datasets import download_weissmann


class GeneNameMapLoader:
    """Create a map from gene name to ensembl id using the weissmann dataset.

    Parameters
    ----------
    data_directory : path to directory where to store the data.
    """

    def __init__(self, data_directory: str):
        self.data_directory = data_directory

    def load(self) -> Dict[str, str]:
        path_k562 = download_weissmann.download_weissmann_k562(self.data_directory)
        path_rpe1 = download_weissmann.download_weissmann_rpe1(self.data_directory)
        data_k562 = sc.read(path_k562)
        intervened_genes_1 = set(data_k562.obs["gene_id"])
        data_rpe1 = sc.read(path_rpe1)
        intervened_genes_2 = set(data_rpe1.obs["gene_id"])
        intervened_genes_total = intervened_genes_1 | intervened_genes_2
        name_to_id_1 = set(zip(data_k562.var["gene_name"], data_k562.var_names))
        name_to_id_2 = set(zip(data_rpe1.var["gene_name"], data_rpe1.var_names))
        name_to_id_total = name_to_id_1 | name_to_id_2
        map = dict(name_to_id_total)
        return {k: v for k, v in map.items() if v in intervened_genes_total}
