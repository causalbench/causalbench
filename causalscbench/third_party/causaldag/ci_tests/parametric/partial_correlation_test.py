"""

License
=======

CausalDAG is distributed with the 3-clause BSD license.

::
Copyright 2018, CausalDAG Developers
Chandler Squires <csquires@mit.edu>

Redistribution and use in source and binary forms, with or 
without modification, are permitted provided that the following
conditions are met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above 
copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided
with the distribution.

3. Neither the name of the copyright holder nor the names of
its contributors may be used to endorse or promote products
derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL 
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""


from typing import Dict
from math import erf
from numpy import linalg, sqrt, log1p, abs, ix_, diag, corrcoef, errstate, cov, mean
from numpy.linalg import inv, pinv


__all__ = [
    "partial_correlation_test",
    "compute_partial_correlation"
]


def numba_inv(A):
    return inv(A)


def compute_partial_correlation(suffstat, i, j, cond_set=None):
    """
    Compute the partial correlation between i and j given ``cond_set``.

    Parameters
    ----------
    suffstat:
        dictionary containing:
        'n' -- number of samples
        'C' -- correlation matrix
        'K' (optional) -- inverse correlation matrix
        'rho' (optional) -- partial correlation matrix (K, normalized so diagonals are 1).
    i:
        position of first variable in correlation matrix.
    j:
        position of second variable in correlation matrix.
    cond_set:
        positions of conditioning set in correlation matrix.

    Returns
    -------
    float
        partial correlation
    """
    C = suffstat.get('C')
    p = C.shape[0]
    rho = suffstat.get('rho')
    K = suffstat.get('K')

    # === COMPUTE PARTIAL CORRELATION
    # partial correlation is correlation if there is no conditioning
    if cond_set is None or len(cond_set) == 0:
        r = C[i, j]
    # used closed-form
    elif len(cond_set) == 1:
        k = list(cond_set)[0]
        r = (C[i, j] - C[i, k]*C[j, k]) / sqrt((1 - C[j, k]**2) * (1 - C[i, k]**2))
    # when conditioning on everything, partial correlation comes from normalized precision matrix
    elif len(cond_set) == p - 2 and rho is not None:
        r = -rho[i, j]
    # faster to use Schur complement if conditioning set is large and precision matrix is pre-computed
    elif len(cond_set) >= p/2 and K is not None:
        rest = list(set(range(C.shape[0])) - {i, j, *cond_set})

        if len(rest) == 1:
            theta_ij = K[ix_([i, j], [i, j])] - K[ix_([i, j], rest)] @ K[ix_(rest, [i, j])] / K[rest[0], rest[0]]
        else:
            if linalg.det(K[ix_(rest, rest)]) != 0:
                theta_ij = K[ix_([i, j], [i, j])] - K[ix_([i, j], rest)] @ pinv(K[ix_(rest, rest)]) @ K[ix_(rest, [i, j])] 
            else:
                print('Not invertible incident with gene indices: ' +str(i) +' ' +str(j))
                r = 0
                return r
        r = -theta_ij[0, 1] / sqrt(theta_ij[0, 0] * theta_ij[1, 1])
    else:
        if linalg.det(C[ix_([i, j, *cond_set], [i, j, *cond_set])]) != 0:
            theta = pinv(C[ix_([i, j, *cond_set], [i, j, *cond_set])])  # TODO: what to do if not invertible?
            r = -theta[0, 1]/sqrt(theta[0, 0] * theta[1, 1])
        else:
            print('Not invertible incident with gene indices: ' +str(i) +' ' +str(j))
            r = 0

    return r


def partial_correlation_test(suffstat: Dict, i, j, cond_set=None, alpha=None):
    """
    Test the null hypothesis that i and j are conditionally independent given ``cond_set``.

    Uses Fisher's z-transform.

    Parameters
    ----------
    suffstat:
        dictionary containing:

        * ``n`` -- number of samples
        * ``C`` -- correlation matrix
        * ``K`` (optional) -- inverse correlation matrix
        * ``rho`` (optional) -- partial correlation matrix (K, normalized so diagonals are 1).
    i:
        position of first variable in correlation matrix.
    j:
        position of second variable in correlation matrix.
    cond_set:
        positions of conditioning set in correlation matrix.
    alpha:
        Significance level.

    Returns
    -------
    dict
        dictionary containing:

        * ``statistic``
        * ``p_value``
        * ``reject``
    """
    n = suffstat['n']
    n_cond = 0 if cond_set is None else len(cond_set)
    alpha = 1/n if alpha is None else alpha

    r = compute_partial_correlation(suffstat, i, j, cond_set=cond_set)

    # === COMPUTE STATISTIC AND P-VALUE
    # note: log1p(2r/(1-r)) = log((1+r)/(1-r)) but is more numerically stable for r near 0
    # r = 1 causes warnings but gives the correct answer
    with errstate(divide='ignore', invalid='ignore'):
        statistic = sqrt(n - n_cond - 3) * abs(.5 * log1p(2*r/(1 - r)))
    # note: erf is much faster than norm.cdf
    p_value = 2*(1 - .5*(1 + erf(statistic/sqrt(2))))

    return dict(statistic=statistic, p_value=p_value, reject=p_value < alpha)