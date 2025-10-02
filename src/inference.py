"""Module for selective inference."""

import numpy as np
from sicore import (  # type: ignore[import]
    RandomizedSelectiveInference,
    SelectiveInferenceNorm,
)

from src.ms import MarginalScreening


def sample_full_covariance(dim: int, rng: np.random.Generator) -> np.ndarray:
    L = np.zeros((dim, dim))
    L[0, 0] = 1.0
    for k in range(1, dim):
        alpha = 1.5 - 1.0 + (dim - k) / 2.0
        y = rng.beta(k / 2.0, alpha)
        u = rng.normal(size=k)
        u /= np.linalg.norm(u)
        w = np.sqrt(y) * u
        L[k, :k] = w
        L[k, k] = np.sqrt(1.0 - y)
    stds = rng.uniform(0.8, 1.2, size=dim)
    return np.diag(stds) @ L @ L.T @ np.diag(stds)


def sample_diagonal_covariance(dim: int, rng: np.random.Generator) -> np.ndarray:
    stds = rng.uniform(0.6, 1.4, size=dim)
    return stds**2


def sample_identity_covariance(dim: int, rng: np.random.Generator) -> float:
    _ = dim
    return float(rng.uniform(0.6, 1.4)) ** 2.0


def make_data(
    rng: np.random.Generator,
    n: int,
    d: int,
    delta: float,
    tau: float,
    cov_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float | np.ndarray]:
    cov: float | np.ndarray
    match cov_type:
        case "base":
            cov = 1.0
            noise = rng.normal(size=n, scale=1.0)
        case "full":
            cov = sample_full_covariance(n, rng)
            noise = rng.multivariate_normal(mean=np.zeros(n), cov=cov)
        case "diag":
            cov = sample_diagonal_covariance(n, rng)
            noise = rng.normal(size=n, scale=1.0) * np.sqrt(cov)
        case "identity":
            cov = sample_identity_covariance(n, rng)
            noise = rng.normal(size=n, scale=np.sqrt(cov))

    beta = delta * np.ones(d)
    X = rng.normal(size=(n, d))
    y = X @ beta + noise
    omega = rng.normal(size=n, scale=tau)
    return X, y, omega, cov


def randomized_inference(
    rng: np.random.Generator,
    n: int = 100,
    d: int = 10,
    delta: float = 0.0,
    k: int = 3,
    cov_type: str = "base",
    tau: float = 1.0,
):
    X, y, omega, cov = make_data(rng, n, d, delta, tau, cov_type)

    ms = MarginalScreening(X, y + omega, k)
    eta = ms.construct_eta(rng.integers(k))
    si = RandomizedSelectiveInference(
        y,
        cov,
        omega,
        tau**2,
        eta,
    )
    result = si.inference(ms.algorithm, ms.model_selector, confidence_level=0.9)
    p_randomized = result.p_value
    ci_lower, ci_upper = result.confidence_interval
    mle = result.point_estimate

    true_signal = delta * eta @ (X @ np.ones(d))
    is_contain = (ci_lower <= true_signal) and (true_signal <= ci_upper)

    return true_signal, p_randomized, [ci_lower, ci_upper], is_contain, mle


def polyhedral_inference(
    rng: np.random.Generator,
    n: int = 100,
    d: int = 10,
    delta: float = 0.0,
    k: int = 3,
    cov_type: str = "base",
):
    X, y, _, cov = make_data(rng, n, d, delta, 1.0, cov_type)

    ms = MarginalScreening(X, y, k)
    eta = ms.construct_eta(rng.integers(k))
    si = SelectiveInferenceNorm(y, cov, eta)
    result = si.inference(ms.algorithm, ms.model_selector, inference_mode="exhaustive")

    p_value = result.p_value

    ci_lower, ci_upper = si.interval_estimate(result, confidence_level=0.9)

    true_signal = delta * eta @ (X @ np.ones(d))
    is_contain = (ci_lower <= true_signal) and (true_signal <= ci_upper)

    mle = si.point_estimate(result)
    return true_signal, p_value, [ci_lower, ci_upper], is_contain, mle


def naive_inference(
    rng: np.random.Generator,
    n: int = 100,
    d: int = 10,
    delta: float = 0.0,
    k: int = 3,
    cov_type: str = "base",
):
    X, y, _, cov = make_data(rng, n, d, delta, 1.0, cov_type)

    ms = MarginalScreening(X, y, k)
    eta = ms.construct_eta(rng.integers(k))
    si = SelectiveInferenceNorm(y, cov, eta)
    result = si.inference(
        ms.algorithm,
        ms.model_selector,
        inference_mode="over_conditioning",
    )

    p_value = result.naive_p_value()
    ci_lower, ci_upper = (
        result.stat - result.null_rv.ppf(0.95),
        result.stat + result.null_rv.ppf(0.95),
    )

    mle = result.stat

    true_signal = delta * eta @ (X @ np.ones(d))
    is_contain = (ci_lower <= true_signal) and (true_signal <= ci_upper)

    return true_signal, p_value, [ci_lower, ci_upper], is_contain, mle
