"""
Copyright (C) 2023  GlaxoSmithKline plc - Mathieu Chevalley;

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

import json
import os
import random

import pandas as pd
import slingpy as sp
from causalscbench.data_access.create_dataset import CreateDataset
from causalscbench.data_access.create_evaluation_datasets import \
    CreateEvaluationDatasets
from causalscbench.data_access.utils.splitting import DatasetSplitter
from causalscbench.evaluation import (biological_evaluation,
                                      statistical_evaluation)


class EvalApp:
    def __init__(
        self,
        output_directory: str,
        data_directory: str,
        metric_file_name: str
    ):
        """
        Evaluate outputs in output_director.

        Args:
            output_directory (str): Directory for output results
            data_directory (str): Directory to store the datasets
        """
        self.data_directory = data_directory
        self.output_directory = output_directory
        self.metric_file_name = metric_file_name
        self.dataset_splitter_k562 = None
        self.dataset_splitter_rpe1 = None
        self.corum_evaluator = None
        self.lr_evaluator = None
        self.quantitative_evaluator = None
        self.chipseq_evaluator_k562 = None
        self.chipseq_evaluator_rpe1 = None

    def load_data(self):
        path_k562, path_rpe1 = CreateDataset(self.data_directory).load()

        self.dataset_splitter_k562 = DatasetSplitter(path_k562)
        self.dataset_splitter_rpe1 = DatasetSplitter(path_rpe1)

    def load_evaluators(self):
        (
            corum,
            lr_pairs,
            string_network_pairs,
            string_physical_pairs,
            chipseq_pairs_k562
        ) = CreateEvaluationDatasets(self.data_directory, "weissmann_k562").load()
        (
            corum,
            lr_pairs,
            string_network_pairs,
            string_physical_pairs,
            chipseq_pairs_rpe1
        ) = CreateEvaluationDatasets(self.data_directory, "weissmann_rpe1").load()
        self.corum_evaluator = biological_evaluation.Evaluator(corum)
        self.lr_evaluator = biological_evaluation.Evaluator(lr_pairs)
        self.string_network_evaluator = biological_evaluation.Evaluator(
            string_network_pairs
        )
        self.string_physical_evaluator = biological_evaluation.Evaluator(
            string_physical_pairs
        )
        self.chipseq_evaluator_k562 = biological_evaluation.Evaluator(
            chipseq_pairs_k562
        )
        self.chipseq_evaluator_rpe1 = biological_evaluation.Evaluator(
            chipseq_pairs_rpe1
        )
        (
            expression_matrix_test,
            interventions_test,
            gene_names,
        ) = self.dataset_splitter_k562.get_test_data()
        self.quantitative_evaluator_k562 = statistical_evaluation.Evaluator(
            expression_matrix_test, interventions_test, gene_names
        )
        (
            expression_matrix_test,
            interventions_test,
            gene_names,
        ) = self.dataset_splitter_rpe1.get_test_data()
        self.quantitative_evaluator_rpe1 = statistical_evaluation.Evaluator(
            expression_matrix_test, interventions_test, gene_names
        )

    def evaluate(self):
        dirs = os.listdir(self.output_directory)
        random.shuffle(dirs)
        for d in dirs:
            arg_dir = os.path.join(self.output_directory, d, "arguments.json")
            network_dir = os.path.join(self.output_directory, d, "output_network.csv")
            if (
                not os.path.exists(arg_dir)
                or not os.path.exists(network_dir)
                or os.path.exists(
                    os.path.join(self.output_directory, d, self.metric_file_name)
                )
            ):
                print("continue")
                continue

            with open(arg_dir) as f:
                data = json.load(f)
                dataset = data["dataset_name"]

            output_network_pd = pd.read_csv(network_dir, index_col=0, skiprows=0)
            output_network = [(b[0], b[1]) for _, b in output_network_pd.iterrows()]

            corum_evaluation = self.corum_evaluator.evaluate_network(output_network)
            ligand_receptor_evaluation = self.lr_evaluator.evaluate_network(
                output_network
            )
            string_network_evaluation = self.string_network_evaluator.evaluate_network(
                output_network
            )
            string_physical_evaluation = (
                self.string_physical_evaluator.evaluate_network(output_network)
            )
            if dataset == "weissmann_k562":
                quantitative_test_evaluation = (
                    self.quantitative_evaluator_k562.evaluate_network(output_network, max_path_length=1)
                )
                chipseq_evaluation = self.chipseq_evaluator_k562.evaluate_network(output_network, directed=True)
            else:
                quantitative_test_evaluation = (
                    self.quantitative_evaluator_rpe1.evaluate_network(output_network, max_path_length=1)
                )
                chipseq_evaluation = self.chipseq_evaluator_rpe1.evaluate_network(output_network, directed=True)
            metrics = {
                "corum_evaluation": corum_evaluation,
                "ligand_receptor_evaluation": ligand_receptor_evaluation,
                "quantitative_test_evaluation": quantitative_test_evaluation,
                "string_network_evaluation": string_network_evaluation,
                "string_physical_evaluation": string_physical_evaluation,
                "chipseq_evaluation": chipseq_evaluation,
            }
            with open(
                os.path.join(self.output_directory, d, self.metric_file_name), "w"
            ) as output:
                json.dump(metrics, output)

    def run(self):
        self.load_data()
        self.load_evaluators()
        self.evaluate()


def main():
    app = sp.instantiate_from_command_line(EvalApp)
    results = app.run()


if __name__ == "__main__":
    main()
