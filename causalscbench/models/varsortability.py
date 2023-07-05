"""
MIT License

Copyright (c) 2021 Alexander Reisach, Sebastian Weichwald

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import List, Tuple
import numpy as np
from causalscbench.models.abstract_model import AbstractInferenceModel
from causalscbench.models.training_regimes import TrainingRegime
from sklearn.linear_model import LinearRegression, LassoLarsIC


class Sortnregress(AbstractInferenceModel):
    """
    Simple algorithm based on marginal variance to recover the causal order
    from https://proceedings.neurips.cc/paper/2021/file/e987eff4a7c7b7e580d659feb6f60c1a-Paper.pdf
    """
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
        if not training_regime == TrainingRegime.Observational:
            return []

        LR = LinearRegression()
        LL = LassoLarsIC(criterion='bic')

        d = len(gene_names)
        W = np.zeros((d, d))
        increasing = np.argsort(np.var(expression_matrix, axis=0))

        for k in range(1, d):
            covariates = increasing[:k]
            target = increasing[k]

            LR.fit(expression_matrix[:, covariates], expression_matrix[:, target].ravel())
            weight = np.abs(LR.coef_)
            LL.fit(expression_matrix[:, covariates] * weight, expression_matrix[:, target].ravel())
            W[covariates, target] = LL.coef_ * weight

        indices = np.transpose(np.nonzero(W))
        edges = set()
        for (i, j) in indices:
            edges.add((gene_names[i], gene_names[j]))
        return list(edges)