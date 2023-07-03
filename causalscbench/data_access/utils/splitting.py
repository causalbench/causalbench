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
from itertools import compress
from typing import List, Tuple

import numpy as np
from sklearn import model_selection


class DatasetSplitter(object):
    """Util class to split the dataset and other training configurations."""

    def __init__(self, dataset_path: str, subset_data: float = 1.0) -> None:
        """
        Args:
            dataset_path: path to dataset as created by data_access/create_dataset.py
            subset_data: whether to subset the training data to the given fraction. Default: 1.0 (full available data)
        """
        with np.load(dataset_path, mmap_mode="r", allow_pickle=True) as arr:
            expression_matrix = arr["expression_matrix"]
            self.gene_names = list(arr["var_names"])
            interventions = arr["interventions"]

        (
            self.expression_matrix_train,
            self.expression_matrix_test,
            self.interventions_train,
            self.interventions_test,
        ) = model_selection.train_test_split(
            expression_matrix,
            interventions,
            test_size=0.2,
            random_state=0,
            stratify=interventions,
        )
        if subset_data < 1.0:
            (
                self.expression_matrix_train,
                _,
                self.interventions_train,
                _,
            ) = model_selection.train_test_split(
                self.expression_matrix_train,
                self.interventions_train,
                train_size=subset_data,
                random_state=0,
                stratify=self.interventions_train,
            )

    def get_test_data(self) -> Tuple[np.array, List[str], List[str]]:
        return self.expression_matrix_test, self.interventions_test, self.gene_names

    def get_observational(self) -> Tuple[np.array, List[str], List[str]]:
        """
        Returns:
            Tuple: dataset with only observational samples.
        """
        return self.get_partial_interventional(fraction=0.0)

    def get_partial_interventional(
        self, fraction: float, seed: int = 0
    ) -> Tuple[np.array, List[str], List[str]]:
        """Only keep the interventional data for a subset of the genes

        Args:
            fraction: the fraction of samples of intervened genes to keep
            seed: seed for reproducibility of the gene random selection

        Returns:
            Tuple: result dataset
        """
        intervened_genes = list(dict.fromkeys(self.interventions_train))
        nb_to_keep = int(fraction * len(intervened_genes))
        random.seed(seed)
        genes_to_keep = set(random.sample(intervened_genes, nb_to_keep))
        genes_to_keep.add("non-targeting")
        samples_to_keep = [x in genes_to_keep for x in self.interventions_train]
        return (
            self.expression_matrix_train[samples_to_keep, :],
            compress(self.interventions_train, samples_to_keep),
            self.gene_names,
        )

    def get_interventional(self) -> Tuple[np.array, List[str], List[str]]:
        """
        Returns:
            Tuple: dataset with interventions for all genes.
        """
        return self.get_partial_interventional(fraction=1.0)
