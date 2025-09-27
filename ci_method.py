"""Module for Comparing Confidence Intervals."""

from itertools import product

import numpy as np
import polars as pl  # type: ignore[import]
from joblib import Parallel, delayed  # type: ignore[import]
from scipy.stats import beta, norm  # type: ignore[import]


def clopper_pearson_interval(
    k: int,
    n: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    lower_bound = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    upper_bound = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0

    return (lower_bound, upper_bound)


def wilson_score_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    p_hat = k / n
    z = norm.ppf(1 - alpha / 2)
    denominator = 1 + (z**2 / n)
    center_adjusted_probability = p_hat + (z**2 / (2 * n))
    adjusted_standard_deviation = np.sqrt((p_hat * (1 - p_hat) + (z**2 / (4 * n))) / n)

    lower_bound = (
        center_adjusted_probability - z * adjusted_standard_deviation
    ) / denominator
    upper_bound = (
        center_adjusted_probability + z * adjusted_standard_deviation
    ) / denominator
    lower_bound = lower_bound if k > 0 else 0.0
    upper_bound = upper_bound if k < n else 1.0

    return (max(0.0, lower_bound), min(1.0, upper_bound))


def agresti_coull_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    z = norm.ppf(1 - alpha / 2)
    n_tilde = n + z**2
    p_tilde = (k + (z**2 / 2)) / n_tilde
    margin_of_error = z * np.sqrt((p_tilde * (1 - p_tilde)) / n_tilde)

    lower_bound = p_tilde - margin_of_error
    upper_bound = p_tilde + margin_of_error

    return (max(0.0, lower_bound), min(1.0, upper_bound))


def wald_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    p_hat = k / n
    z = norm.ppf(1 - alpha / 2)
    margin_of_error = z * np.sqrt((p_hat * (1 - p_hat)) / n)

    lower_bound = p_hat - margin_of_error
    upper_bound = p_hat + margin_of_error

    return (max(0.0, lower_bound), min(1.0, upper_bound))


def jeffreys_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lower_bound = beta.ppf(alpha / 2, k + 0.5, n - k + 0.5) if k > 0 else 0.0
    upper_bound = beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5) if k < n else 1.0

    return (lower_bound, upper_bound)


def conduct(
    true_p: float,
    method: str,
    n: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    alpha: float = 0.05
    iter: int = 100_000

    p_list = rng.uniform(size=(n, iter))
    k_s = np.count_nonzero(p_list < true_p, axis=0)
    bool_list = []
    length_list = []
    for k in k_s:
        lower, upper = METHODS_DICT[method](k, n, alpha)
        bool_list.append(lower <= true_p <= upper)
        length_list.append(upper - lower)
    return float(np.mean(bool_list)), float(np.mean(length_list))


METHODS_DICT = {
    "clopper_pearson": clopper_pearson_interval,
    "wilson_score": wilson_score_interval,
    "agresti_coull": agresti_coull_interval,
    "wald": wald_interval,
    "jeffreys": jeffreys_interval,
}

if __name__ == "__main__":
    METHODS = list(METHODS_DICT.keys())
    NS = [1000, 2000, 10000]
    TRUE_PS = [0.01 * i for i in range(0, 101)]

    rng = np.random.default_rng(0)

    parameters = list(product(METHODS, NS, TRUE_PS))

    # rewrite with tqdm
    results = Parallel(n_jobs=48)(
        delayed(conduct)(true_p, method, n, rng_)
        for (method, n, true_p), rng_ in zip(parameters, rng.spawn(len(parameters)))
    )

    frame = pl.DataFrame(
        {
            "method": [method for (method, n, true_p) in parameters],
            "n": [n for (method, n, true_p) in parameters],
            "true_p": [true_p for (method, n, true_p) in parameters],
            "coverage": [coverage for (coverage, length) in results],
            "length": [length for (coverage, length) in results],
        }
    )
    frame.write_csv("ci_method_results.csv")
