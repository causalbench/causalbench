from .parametric import *


def get_ci_tester(
        samples,
        test="partial_correlation",
        memoize=False,
        **kwargs
):
    if test == "partial_correlation":
        ci_test = partial_correlation_test
        suffstat = partial_correlation_suffstat(samples)
    else:
        raise ValueError()

    if memoize:
        return MemoizedCI_Tester(ci_test, suffstat, **kwargs)
