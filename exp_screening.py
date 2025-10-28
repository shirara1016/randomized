"""Module for experiments."""

import argparse

import numpy as np
import polars as pl
from joblib import Parallel, delayed  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

from src.inference import screening

N_JOBS = 48


def simulate(
    delta: float,
    base_rng: np.random.Generator,
) -> None:
    n, d = 100, 10
    k = 3

    split_rates = [(i + 1) / 100 for i in range(100)]
    randomized_rates = []
    splitting_rates = []
    for split_rate in tqdm(split_rates):
        randomized_results = Parallel(n_jobs=N_JOBS)(
            delayed(screening)(
                rng,
                n=n,
                d=d,
                delta=delta,
                k=k,
                split_rate=split_rate,
                mode="randomized",
            )
            for rng in base_rng.spawn(1000)
        )
        splitting_results = Parallel(n_jobs=N_JOBS)(
            delayed(screening)(
                rng,
                n=n,
                d=d,
                delta=delta,
                k=k,
                split_rate=split_rate,
                mode="splitting",
            )
            for rng in base_rng.spawn(1000)
        )
        randomized_rate = sum(randomized_results) / len(randomized_results)
        splitting_rate = sum(splitting_results) / len(splitting_results)
        randomized_rates.append(randomized_rate)
        splitting_rates.append(splitting_rate)

    frame = pl.DataFrame(
        {
            "split_rate": split_rates,
            "randomized_rate": randomized_rates,
            "splitting_rate": splitting_rates,
        }
    )
    frame.write_csv(f"results/screening_delta_{delta:.1f}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delta",
        type=float,
        default=0.5,
    )
    args = parser.parse_args()
    base_rng = np.random.default_rng(0)
    simulate(args.delta, base_rng)
