"""Module for experiments."""

import argparse

import numpy as np
import polars as pl
from joblib import Parallel, delayed  # type: ignore[import]
from tqdm import tqdm  # type: ignore[import]

from src.inference import randomized_inference, splitting_inference

N_JOBS = 50


def simulate(
    delta: float,
    base_rng: np.random.Generator,
) -> None:
    n, d = 100, 10
    k = 3

    split_rates = [(i + 1) / 100 for i in range(9, 100)]
    randomized_powers = []
    splitting_powers = []
    for split_rate in tqdm(split_rates):
        randomized_results = Parallel(n_jobs=N_JOBS)(
            delayed(randomized_inference)(
                rng,
                n=n,
                d=d,
                delta=delta,
                k=k,
                tau=((1.0 - split_rate) / split_rate) ** 0.5,
            )
            for rng in base_rng.spawn(1000)
        )
        splitting_results = Parallel(n_jobs=N_JOBS)(
            delayed(splitting_inference)(
                rng,
                n=n,
                d=d,
                delta=delta,
                k=k,
                split_rate=split_rate,
            )
            for rng in base_rng.spawn(1000)
        )
        randomized_power = np.mean([res[1] < 0.05 for res in randomized_results])
        splitting_power = np.mean([res[1] < 0.05 for res in splitting_results])
        randomized_powers.append(randomized_power)
        splitting_powers.append(splitting_power)

    frame = pl.DataFrame(
        {
            "split_rate": split_rates,
            "randomized_power": randomized_powers,
            "splitting_power": splitting_powers,
        }
    )
    frame.write_csv(f"results/screening_delta_{delta:.1f}_power.csv")


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
