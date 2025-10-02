"""Module for experiments."""

import argparse

import numpy as np
import polars as pl
from joblib import Parallel, delayed  # type: ignore[import]

from src.inference import naive_inference, polyhedral_inference, randomized_inference


def simulate(
    method: str,
    delta: float,
    cov_type: str,
    base_rng: np.random.Generator,
) -> None:
    n, d = 100, 10
    k = 3
    tau = 1.0

    match method:
        case "randomized":
            results = Parallel(n_jobs=32)(
                delayed(randomized_inference)(
                    rng,
                    n=n,
                    d=d,
                    delta=delta,
                    k=k,
                    tau=tau,
                    cov_type=cov_type,
                )
                for rng in base_rng.spawn(10000)
            )
        case "polyhedral":
            results = Parallel(n_jobs=32)(
                delayed(polyhedral_inference)(
                    rng,
                    n=n,
                    d=d,
                    delta=delta,
                    k=k,
                    cov_type=cov_type,
                )
                for rng in base_rng.spawn(10000)
            )
        case "naive":
            results = Parallel(n_jobs=32)(
                delayed(naive_inference)(
                    rng,
                    n=n,
                    d=d,
                    delta=delta,
                    k=k,
                    cov_type=cov_type,
                )
                for rng in base_rng.spawn(10000)
            )
    true_signal_list, p_list, ci_list, is_contain_list, mle_list = zip(*results)
    ci_lower_list, ci_upper_list = zip(*ci_list)

    frame = pl.DataFrame(
        {
            "method": method,
            "delta": delta,
            "true_signal": true_signal_list,
            "p": p_list,
            "ci_lower": ci_lower_list,
            "ci_upper": ci_upper_list,
            "is_contain": is_contain_list,
            "mle": mle_list,
        }
    )
    frame.write_csv(f"results/simulation_{cov_type}_{method}_{delta}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        type=str,
        default="randomized",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--cov",
        type=str,
        default="base",
    )
    args = parser.parse_args()

    sub_seed = (1 if args.method == "randomized" else 0) + (
        2 if args.method == "naive" else 0
    )
    seed = int(args.delta * 1000) + sub_seed
    base_rng = np.random.default_rng(seed + 3)
    simulate(args.method, args.delta, args.cov, base_rng)
