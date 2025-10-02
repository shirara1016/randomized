"""Module for selective inference for marginal screening."""

import numpy as np
from sicore import (  # type: ignore[import]
    RealSubset,
    linear_polynomials_below_zero,
)


class MarginalScreening:
    """A class for marginal screening."""

    def __init__(self, X: np.ndarray, y: np.ndarray, k: int) -> None:
        self.X, self.y, self.k = X, y, k
        self.M = self._feature_selection(X, y, k)

    def _feature_selection(self, X: np.ndarray, y: np.ndarray, k: int) -> list[int]:
        return np.argsort(np.abs(X.T @ y))[::-1][:k].tolist()

    def algorithm(
        self,
        a: np.ndarray,
        b: np.ndarray,
        z: float,
    ) -> tuple[list[int], RealSubset]:
        a, b = self.X.T @ a, self.X.T @ b

        signs = np.sign(a + b * z)
        intervals_ = linear_polynomials_below_zero(-signs * a, -signs * b)
        intervals = RealSubset(intervals_)
        a, b = signs * a, signs * b

        collerations = a + b * z
        indexes = np.argsort(collerations)[::-1]

        active_set = indexes[: self.k]
        inactive_set = indexes[self.k :]

        for active in active_set:
            temp_intervals = linear_polynomials_below_zero(
                a[inactive_set] - a[active],
                b[inactive_set] - b[active],
            )
            intervals = intervals & RealSubset(temp_intervals)

        if z not in intervals:
            raise ValueError
        return indexes[: self.k].tolist(), intervals

    def model_selector(self, M: list[int]) -> bool:
        return set(self.M) == set(M)

    def construct_eta(self, index: int) -> np.ndarray:
        return (
            self.X[:, self.M]
            @ np.linalg.inv(self.X[:, self.M].T @ self.X[:, self.M])[:, index]
        )
