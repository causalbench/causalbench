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

from typing import List, Tuple

import numpy as np
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


class LassoFeatureSelection(AbstractInferenceModel):
    """Network inference model based on the LASSO model.
    For each variable, a LASSO regressor is fit and the most predictive variables are set as parents."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        edges = set()
        for i in range(len(gene_names)):
            selector = np.full(len(gene_names), True)
            selector[i] = False
            X = expression_matrix[:, selector]
            y = expression_matrix[:, i]
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            parents_selector = SelectFromModel(
                estimator=Lasso(alpha=0.05, random_state=seed), max_features=40
            ).fit(X, y)
            parents = gene_names[selector][parents_selector.get_support()]
            for parent in parents:
                edges.add((parent, gene_names[i]))
        return list(edges)


class RandomForestFeatureSelection(AbstractInferenceModel):
    """Network inference model based on the Random Forest model.
    For each variable, a Random Forest regressor is fit and the most predictive variables are set as parents."""

    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self,
        expression_matrix: np.array,
        interventions: List[str],
        gene_names: List[str],
        training_regime: TrainingRegime,
        seed: int = 0,
    ) -> List[Tuple]:
        edges = set()
        for i in range(len(gene_names)):
            selector = np.full(len(gene_names), True)
            selector[i] = False
            X = expression_matrix[:, selector]
            y = expression_matrix[:, i]
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            parents_selector = SelectFromModel(
                estimator=RandomForestRegressor(
                    n_estimators=100, n_jobs=-1, random_state=seed
                ),
                max_features=40,
            ).fit(X, y)
            parents = gene_names[selector][parents_selector.get_support()]
            for parent in parents:
                edges.add((parent, gene_names[i]))
        return list(edges)
