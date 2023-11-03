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
import time

import pandas as pd
import slingpy as sp
from  slingpy.utils import logging
from causalscbench.apps.utils.run_utils import (
    create_experiment_folder, get_if_valid_custom_function_file)
from causalscbench.data_access.create_dataset import CreateDataset
from causalscbench.data_access.create_evaluation_datasets import \
    CreateEvaluationDatasets
from causalscbench.data_access.utils.splitting import DatasetSplitter
from causalscbench.evaluation import (biological_evaluation,
                                      statistical_evaluation)
from causalscbench.models import training_regimes
from causalscbench.models.arboreto_baselines import GENIE, GRNBoost
from causalscbench.models.causallearn_models import GES, PC
from causalscbench.models.dcdi_models import DCDI, DCDFG
from causalscbench.models.feature_selection import (
    LassoFeatureSelection, RandomForestFeatureSelection)
from causalscbench.models.gies import GIES
from causalscbench.models.notears import NotearsLin, NotearsMLP
from causalscbench.models.random_network import FullyConnected, RandomWithSize
from causalscbench.models.sparsest_permutations import (
    GreedySparsestPermutation,
    InterventionalGreedySparsestPermutation,
)
from causalscbench.models.varsortability import Sortnregress

DATASET_NAMES = [
    "weissmann_k562",
    "weissmann_rpe1",
]

METHODS = [
    "random100",
    "random1000",
    "random10000",
    "fully-connected",
    "lasso",
    "random_forest",
    "grnboost",
    "genie",
    "ges",
    "gies",
    "pc",
    "mvpc",
    "gsp",
    "igsp",
    "notears-lin",
    "notears-lin-sparse",
    "notears-mlp",
    "notears-mlp-sparse",
    "DCDI-G",
    "DCDI-DSF",
    "DCDFG-LIN",
    "DCDFG-MLP",
    "corum",
    "lr",
    "string_network",
    "string_physical",
    "custom",
    "chipseq",
    "pooled_biological_networks",
    "sortnregress",
]


class MainApp:
    def __init__(
        self,
        output_directory: str,
        data_directory: str,
        model_name: str = METHODS[0],
        inference_function_file_path: str = "",
        dataset_name: str = DATASET_NAMES[0],
        model_seed: int = 0,
        training_regime: training_regimes.TrainingRegime = training_regimes.TrainingRegime.Interventional,
        partial_intervention_seed: int = 0,
        fraction_partial_intervention: float = 1.0,
        subset_data: float = 1.0,
        exp_id: str = "",
        max_path_length: int = -1,
        omission_estimation_size: int = 0,
        filter: bool = False,
    ):
        """
        Main full training pipeline.

        Args:
            output_directory (str): Directory for output results
            data_directory (str): Directory to store the datasets
            model_name (str, optional): Which method to run. Defaults to METHODS[0].
            inference_function_file_path (str, optional): Path to file for custom inference function. Default to empty string.
            dataset_name (List[str], optional): Which dataset to use. Defaults to DATASET_NAMES[0].
            model_seed (int, optional): Seed for model reproducibility. Defaults to 0.
            training_regime (training_regimes.TrainingRegime, optional): Choice of training regime. Defaults to training_regimes.Interventional.
            partial_intervention_seed (int, optional): If training_regime is partial intervention, seed for random selection of perturbed genes. Defaults to 0.
            fraction_partial_intervention (float, optional):  If training_regime is partial intervention, fraction of genes which should have interventional data. Defaults to 1.0.
            subset_data (float, optional): Option to subset the whole dataset for easier training. Defaults to 1.0.
            exp_id (str, optional): Unique experiment id (6 digit number). Default to randomly generated.
            max_path_length (int, optional): Maximum length of path to consider for statistical evaluation. Default to -1 (all paths).
            omission_estimation_size (int, optional): Number of negative samples to draw to estimate the false omission rate. If 0, the FOR is not checked. 
        """
        self.data_directory = data_directory
        self.output_directory = create_experiment_folder(exp_id, output_directory)
        self.model_name = model_name
        self.inference_function_file_path = inference_function_file_path
        self.dataset_name = dataset_name
        self.model_seed = model_seed
        self.training_regime = training_regime
        self.partial_intervention_seed = partial_intervention_seed
        self.fraction_partial_intervention = fraction_partial_intervention
        self.subset_data = subset_data
        self.exp_id = exp_id
        self.max_path_length = max_path_length
        self.filter = filter
        self.omission_estimation_size = omission_estimation_size
        self.check_false_omission_rate = omission_estimation_size > 0
        self.model = None
        self.dataset_splitter = None
        self.corum_evaluator = None
        self.lr_evaluator = None
        self.chipseq_evaluator = None
        self.quantitative_evaluator = None
        self.pooled_biological_evaluator = None
        self.pooled_biological_significant_evaluator = None

    def load_model(self):
        models_dict = {
            "random100": RandomWithSize(100),
            "random1000": RandomWithSize(1000),
            "random10000": RandomWithSize(10000),
            "fully-connected": FullyConnected(),
            "lasso": LassoFeatureSelection(),
            "random_forest": RandomForestFeatureSelection(),
            "grnboost": GRNBoost(),
            "genie": GENIE(),
            "ges": GES(),
            "gies": GIES(),
            "pc": PC(missing_value=False),
            "mvpc": PC(missing_value=True),
            "gsp": GreedySparsestPermutation(),
            "igsp": InterventionalGreedySparsestPermutation(),
            "notears-lin": NotearsLin(lambda1=0.0),
            "notears-lin-sparse": NotearsLin(lambda1=0.001),
            "notears-mlp": NotearsMLP(lambda1=0.0),
            "notears-mlp-sparse": NotearsMLP(lambda1=0.001),
            "DCDI-G": DCDI("DCDI-G"),
            "DCDI-DSF": DCDI("DCDI-DSF"),
            "DCDFG-LIN": DCDFG("linear"),
            "DCDFG-MLP": DCDFG("mlplr"),
            "corum": self.corum_evaluator,
            "lr": self.lr_evaluator,
            "string_network": self.string_network_evaluator,
            "string_physical": self.string_physical_evaluator,
            "chipseq": self.chipseq_evaluator,
            "pooled_biological_networks": self.pooled_biological_evaluator,
            "sortnregress": Sortnregress(),
        }
        if self.model_name not in METHODS:
            raise NotImplementedError()
        if self.model_name == "custom":
            self.model = get_if_valid_custom_function_file(
                self.inference_function_file_path
            )()
        else:
            self.model = models_dict[self.model_name]

    def load_data(self):
        path_k562, path_rpe1 = CreateDataset(self.data_directory, self.filter).load()

        if self.dataset_name == "weissmann_k562":
            self.dataset_splitter = DatasetSplitter(path_k562, self.subset_data)
        elif self.dataset_name == "weissmann_rpe1":
            self.dataset_splitter = DatasetSplitter(path_rpe1, self.subset_data)
        else:
            raise NotImplementedError()

    def load_evaluators(self):
        (
            corum,
            lr_pairs,
            string_network_pairs,
            string_physical_pairs,
            chipseq_pairs,
        ) = CreateEvaluationDatasets(self.data_directory, self.dataset_name).load()
        self.corum_evaluator = biological_evaluation.Evaluator(corum)
        self.lr_evaluator = biological_evaluation.Evaluator(lr_pairs)
        self.string_network_evaluator = biological_evaluation.Evaluator(
            string_network_pairs
        )
        self.string_physical_evaluator = biological_evaluation.Evaluator(
            string_physical_pairs
        )
        self.chipseq_evaluator = biological_evaluation.Evaluator(chipseq_pairs)
        pooled_biological_pairs = corum.union(
            lr_pairs, string_network_pairs, string_physical_pairs, chipseq_pairs
        )
        network_undirected = set()
        for i, j in pooled_biological_pairs:
            network_undirected.add((i, j))
            network_undirected.add((j, i))
        self.pooled_biological_evaluator = biological_evaluation.Evaluator(
            network_undirected
        )
        (
            expression_matrix_test,
            interventions_test,
            gene_names,
        ) = self.dataset_splitter.get_test_data()
        self.quantitative_evaluator = statistical_evaluation.Evaluator(
            expression_matrix_test, interventions_test, gene_names
        )
        output_pooled_biological_evaluator = self.pooled_biological_evaluator(None, None, gene_names, "", 0)
        pooled_biological_network_significant = set()
        for edge in output_pooled_biological_evaluator:
            if self.quantitative_evaluator.evaluate_network([edge], max_path_length=0)["output_graph"]["true_positives"] > 0:
                pooled_biological_network_significant.add(edge)
        self.pooled_biological_significant_evaluator = biological_evaluation.Evaluator(
            pooled_biological_network_significant
        )


    def train_and_evaluate(self):
        if self.training_regime == training_regimes.TrainingRegime.Observational:
            (
                expression_matrix_train,
                interventions_train,
                gene_names,
            ) = self.dataset_splitter.get_observational()
        elif (
            self.training_regime == training_regimes.TrainingRegime.PartialIntervational
        ):
            (
                expression_matrix_train,
                interventions_train,
                gene_names,
            ) = self.dataset_splitter.get_partial_interventional(
                self.fraction_partial_intervention, self.partial_intervention_seed
            )
        else:
            (
                expression_matrix_train,
                interventions_train,
                gene_names,
            ) = self.dataset_splitter.get_interventional()
        arguments = {
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "inference_function_file_path": self.inference_function_file_path,
            "model_seed": self.model_seed,
            "training_regime": self.training_regime.name,
            "partial_intervention_seed": self.partial_intervention_seed,
            "fraction_partial_intervention": self.fraction_partial_intervention,
            "subset_data": self.subset_data,
            "exp_id": self.exp_id,
            "max_path_length": self.max_path_length,
            "omission_estimation_size": self.omission_estimation_size,
            "filter": self.filter,
        }
        with open(os.path.join(self.output_directory, "arguments.json"), "w") as output:
            json.dump(arguments, output)
        start_time = time.time()
        logging.info("Starting model training.")
        output_network = self.model(
            expression_matrix_train,
            list(interventions_train),
            gene_names,
            self.training_regime,
            self.model_seed,
        )
        logging.info("Model training finished.")
        end_time = time.time()
        logging.info("Evaluating model.")
        corum_evaluation = self.corum_evaluator.evaluate_network(output_network)
        ligand_receptor_evaluation = self.lr_evaluator.evaluate_network(output_network)
        string_network_evaluation = self.string_network_evaluator.evaluate_network(
            output_network
        )
        string_physical_evaluation = self.string_physical_evaluator.evaluate_network(
            output_network
        )
        chipseq_evaluation = self.chipseq_evaluator.evaluate_network(
            output_network, directed=True
        )
        pooled_biological_evaluation = (
            self.pooled_biological_evaluator.evaluate_network(output_network)
        )
        pooled_biological_sigificant_evaluation = self.pooled_biological_significant_evaluator.evaluate_network(output_network, directed=True)
        quantitative_test_evaluation = self.quantitative_evaluator.evaluate_network(
            output_network, self.max_path_length, self.check_false_omission_rate, self.omission_estimation_size,
        )
        logging.info("Model evaluation finished.")
        metrics = {
            "corum_evaluation": corum_evaluation,
            "ligand_receptor_evaluation": ligand_receptor_evaluation,
            "quantitative_test_evaluation": quantitative_test_evaluation,
            "string_network_evaluation": string_network_evaluation,
            "string_physical_evaluation": string_physical_evaluation,
            "chipseq_evaluation": chipseq_evaluation,
            "pooled_biological_evaluation": pooled_biological_evaluation,
            "pooled_biological_sigificant_evaluation": pooled_biological_sigificant_evaluation,
            "run_time": end_time - start_time,
        }
        with open(os.path.join(self.output_directory, "metrics.json"), "w") as output:
            json.dump(metrics, output)
        pd.DataFrame(output_network).to_csv(
            os.path.join(self.output_directory, "output_network.csv")
        )

        return metrics

    def run(self):
        logging.info("Loading and preprocessing data.")
        self.load_data()
        logging.info("Loading and processing evaluators.")
        self.load_evaluators()
        logging.info("Loading model.")
        self.load_model()
        self.train_and_evaluate()


def main():
    app = sp.instantiate_from_command_line(MainApp)
    results = app.run()


if __name__ == "__main__":
    main()
