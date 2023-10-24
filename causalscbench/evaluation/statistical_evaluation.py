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
from typing import Dict, List, Tuple

import numpy as np
import scipy


class Evaluator(object):
    def __init__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        p_value_threshold=0.05,
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
        self.gene_names = gene_names

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

    def evaluate_network(self, network: List[Tuple], max_path_length = 3, check_false_omission_rate=False, omission_estimation_size=0) -> Dict:
        """
        Use a non-parametric Mannwhitney rank-sum test to test wether perturbing an upstream does have
        an effect on the downstream children genes. The assumptions is that intervening on a parent gene
        should have an effect on the distribution of the expression of a child gene in the network. Also consider
        the all connected graph with the same evaluation (all pairs such that there is a directed path between them)

        Args:
            network: output network as a list of tuples, where (A, B) indicates that gene A acts on gene B
            max_path_length: maximum length of paths to consider for evaluation. If -1, check paths of any length.
            check_false_omission_rate: whether to check the false omission rate of the predicted graph. 
                                        The false omission rate is defined as (FN / (FN + TN)), FN = false negative and TN = true negative
            omission_estimation_size: how many negative pairs (a pair predicted to have no interaction, i.e, there is no path in the output graph) to draw to estimate the false omission rate

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

        # Test graph with paths of length smaller or equal to max_path_length
        all_path_results = []
        if max_path_length == -1:
            max_path_length = len(self.gene_names)
        for _ in range(max_path_length - 1):
            single_step_deeper_all_connected_network = {
                v: n.union(
                    *[
                        all_connected_network[nn]
                        for nn in n
                        if nn in all_connected_network
                    ]
                )
                for v, n in all_connected_network.items()
            }
            single_step_deeper_all_connected_network = {v: c - {v} for v, c in single_step_deeper_all_connected_network.items()}
            if single_step_deeper_all_connected_network == all_connected_network:
                break
            new_edges = {
                key: single_step_deeper_all_connected_network[key] - all_connected_network[key]
                for key in single_step_deeper_all_connected_network
            }
            (
                true_positive_connected,
                false_positive_connected,
                wasserstein_distances_connected,
            ) = self._evaluate_network(new_edges)
            all_path_results.append({
                "true_positives": true_positive_connected,
                "false_positives": false_positive_connected,
                "wasserstein_distance": {
                    "mean": np.mean(wasserstein_distances_connected)
                },
            })
            all_connected_network = single_step_deeper_all_connected_network

        if check_false_omission_rate:
            edges = set()
            # Draw omission_estimation_size edges from the negative set (edges predicted to have no interaction)
            # to estimate the false omission rate and the associated mean wasserstein distance
            random.seed(0)
            while len(edges) < omission_estimation_size:
                pair = random.sample(range(len(self.gene_names)), 2)
                edge = self.gene_names[pair[0]], self.gene_names[pair[1]]
                if edge[0] in all_connected_network and edge[1] in all_connected_network[edge[0]]:
                    continue
                edges.add(edge)
            network_as_dict = {}
            for a, b in edges:
                network_as_dict.setdefault(a, set()).add(b)
            res_random = self._evaluate_network(network_as_dict)
            false_omission_rate = res_random[0] / omission_estimation_size
            negative_mean_wasserstein = np.mean(res_random[2])
        else:
            false_omission_rate = -1
            negative_mean_wasserstein = -1


        return {
            "output_graph": {
                "true_positives": true_positive,
                "false_positives": false_positive,
                "wasserstein_distance": {"mean": np.mean(wasserstein_distances)},
            },
            "all_path_results": all_path_results,
            "false_omission_rate": false_omission_rate,
            "negative_mean_wasserstein": negative_mean_wasserstein
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
                ranksum_result = scipy.stats.mannwhitneyu(
                    observational_samples, interventional_samples
                )
                wasserstein_distance = scipy.stats.wasserstein_distance(
                    observational_samples, interventional_samples, 
                )
                wasserstein_distances.append(wasserstein_distance)
                p_value = ranksum_result[1]
                if p_value < self.p_value_threshold:
                    # Mannwhitney test rejects the hypothesis that the two distributions are similar
                    # -> parent has an effect on the child
                    true_positive += 1
                else:
                    false_positive += 1
        return true_positive, false_positive, wasserstein_distances
