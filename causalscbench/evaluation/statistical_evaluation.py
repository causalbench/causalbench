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

from typing import Dict, List, Tuple

import numpy as np
import scipy


class Evaluator(object):
    def __init__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        p_value_threshold=0.1,
    ) -> None:
        """
        Evaluation module to quantitatively evaluate a network using held-out data.

        Args:
            expression_matrix: a numpy matrix of expression data of size [nb_samples, nb_genes]
            interventions: a list of size [nb_samples] that indicates which gene has been perturb. "non-targeting" means no gene has been perturbed (observational data)
            gene_names: name of the genes in the expression matrix
            p_value_threshold: threshold for statistical significance, default 0.05
        """
        self.gene_to_index = dict(zip(gene_names, range(len(gene_names))))
        self.gene_to_interventions = dict()
        for i, intervention in enumerate(interventions):
            self.gene_to_interventions.setdefault(intervention, []).append(i)
        self.expression_matrix = expression_matrix
        self.p_value_threshold = p_value_threshold

    def get_observational(self, child: str) -> np.array:
        """
        Return all the samples for gene "child" in cells where there was no perturbations

        Args:
            child: Gene name of child to get samples for

        Returns:
            np.array matrix of corresponding samples
        """
        return self.get_interventional(child, "non-targeting")

    def get_interventional(self, child: str, parent: str) -> np.array:
        """
        Return all the samples for gene "child" in cells where "parent" was perturbed

        Args:
            child: Gene name of child to get samples for
            parent: Gene name of gene that must have been perturbed

        Returns:
            np.array matrix of corresponding samples
        """
        return self.expression_matrix[
            self.gene_to_interventions[parent], self.gene_to_index[child]
        ]

    def evaluate_network(self, network: List[Tuple], max_path_length = 3) -> Dict:
        """
        Use a non-parametric Mannwhitney rank-sum test to test wether perturbing an upstream does have
        an effect on the downstream children genes. The assumptions is that intervening on a parent gene
        should have an effect on the distribution of the expression of a child gene in the network. Also consider
        the all connected graph with the same evaluation (all pairs such that there is a directed path between them)

        Args:
            network: output network as a list of tuples, where (A, B) indicates that gene A acts on gene B
            max_path_length: maximum length of paths to consider for evaluation

        Returns:
            Number of true positive and false positive edges for both original graph and all connected graph
        """
        network_as_dict = {}
        for a, b in network:
            network_as_dict.setdefault(a, set()).add(b)
        true_positive, false_positive, wasserstein_distances = self._evaluate_network(
            network_as_dict
        )

        all_connected_network = {**network_as_dict}

        # Augment graph with paths of length smaller or equal to max_path_length
        for _ in range(max_path_length - 1):
            new_all_connected_network = {
                v: n.union(
                    *[
                        all_connected_network[nn]
                        for nn in n
                        if nn in all_connected_network
                    ]
                )
                for v, n in all_connected_network.items()
            }
            if new_all_connected_network == all_connected_network:
                break
            all_connected_network = new_all_connected_network

        all_connected_network = {v: c - {v} for v, c in all_connected_network.items()}
        if all_connected_network == network_as_dict:
            (
                true_positive_connected,
                false_positive_connected,
                wasserstein_distances_connected,
            ) = true_positive, false_positive, wasserstein_distances
        else: 
            (
                true_positive_connected,
                false_positive_connected,
                wasserstein_distances_connected,
            ) = self._evaluate_network(all_connected_network)

        return {
            "output_graph": {
                "true_positives": true_positive,
                "false_positives": false_positive,
                "wasserstein_distance": {"mean": np.mean(wasserstein_distances)},
            },
            "all_path_output": {
                "true_positives": true_positive_connected,
                "false_positives": false_positive_connected,
                "wasserstein_distance": {
                    "mean": np.mean(wasserstein_distances_connected)
                },
            },
        }

    def _evaluate_network(self, network_as_dict):
        true_positive = 0
        false_positive = 0
        wasserstein_distances = []
        for parent in network_as_dict.keys():
            children = network_as_dict[parent]
            for child in children:
                observational_samples = self.get_observational(child)
                interventional_samples = self.get_interventional(child, parent)
                wasserstein_distance = scipy.stats.wasserstein_distance(
                    observational_samples, interventional_samples
                )
                wasserstein_distances.append(wasserstein_distance)
                fraction_outliers = 1 - (
                    sum(observational_samples > 0.0) / len(observational_samples)
                )
                fraction_outliers *= 0.9
                outliers_to_remove = int(fraction_outliers * len(observational_samples))
                observational_samples = np.sort(observational_samples)[
                    outliers_to_remove:
                ]
                outliers_to_remove = int(
                    fraction_outliers * len(interventional_samples)
                )
                interventional_samples = np.sort(interventional_samples)[
                    outliers_to_remove:
                ]
                ranksum_result = scipy.stats.mannwhitneyu(
                    observational_samples, interventional_samples
                )
                p_value = ranksum_result[1]
                if p_value < self.p_value_threshold:
                    # Mannwhitney test rejects the hypothesis that the two distributions are similar
                    # -> parent has an effect on the child
                    true_positive += 1
                else:
                    false_positive += 1
        return true_positive, false_positive, wasserstein_distances
