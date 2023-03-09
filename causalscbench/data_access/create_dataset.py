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
import os
from typing import Tuple

import numpy as np
import pandas as pd
from causalscbench.data_access.datasets import download_weissmann
from causalscbench.data_access.utils import preprocessing


class CreateDataset:
    """Download and create the necessary dataset.

    Parameters
    ----------
    data_directory : path to directory where to store the data.
    """

    def __init__(self, data_directory: str, filter: bool):
        self.data_directory = data_directory
        self.filter = filter

    def preprocess_and_save(self, dataset_path, summary_stats, filename: str) -> str:
        if self.filter:
            filename += "_filtered"
        output_path = os.path.join(self.data_directory, filename + ".npz")
        if not os.path.exists(output_path):
            dataset = preprocessing.preprocess_dataset(dataset_path, summary_stats)
            with open(output_path, "wb") as file:
                np.savez(
                    file,
                    expression_matrix=dataset[0],
                    var_names=dataset[1],
                    interventions=dataset[2],
                )
        return output_path

    def load(self) -> Tuple[str, str]:
        path_k562 = download_weissmann.download_weissmann_k562(self.data_directory)
        path_rpe1 = download_weissmann.download_weissmann_rpe1(self.data_directory)
        if self.filter:
            path_meta = download_weissmann.download_summary_stats(self.data_directory)
            summary_stats_k562 = pd.read_excel(path_meta, sheet_name='TabB_K562_day6_summary_stat')
            summary_stats_rpe1 = pd.read_excel(path_meta, sheet_name='TabC_RPE1_summary_statistic')
        else:
            summary_stats_k562 = None
            summary_stats_rpe1 = None
        output_path_k562 = self.preprocess_and_save(path_k562, summary_stats_k562, "dataset_k562")
        output_path_rpe1 = self.preprocess_and_save(path_rpe1, summary_stats_rpe1, "dataset_rpe1")
        return output_path_k562, output_path_rpe1
