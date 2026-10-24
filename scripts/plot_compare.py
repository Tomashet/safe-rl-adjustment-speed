import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_curve(run_dir: str, metric: str, step_col: str = "timestep"):
    csv_path = os.path.join("runs", run_dir, "train_monitor.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    if step_col not in df.columns:
        raise ValueError(f"{run_dir}: missing '{step_col}'")
    if metric not in df.columns:
        raise ValueError(f"{run_dir}: missing '{metric}'")
    x = df[step_col].to_numpy(dtype=float)
    y = df[metric].to_numpy(dtype=float)
    return x, y


def resample_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    # Linear interpolation onto a common grid (good for step-indexed logs)
    return np.interp(grid, x, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--groups",
        nargs="+",
        required=True,
        help="Pairs name=pattern. Example: baseline=B_dqn_seed* adj=F_dqn_seed*",
    )
    ap.add_argument("--metric", default="violation", help="e.g., violation, adj_risk, shield_used")
    ap.add_argument("--smooth", type=int, default=50, help="moving average width on resampled grid")
    ap.add_argument("--grid_points", type=int, default=2000, help="number of points on common grid")
    args = ap.parse_args()

    groups = []
    for item in args.groups:
        if "=" not in item:
            raise ValueError("Group must be name=pattern, e.g., baseline=B_dqn_seed*")
        name, pattern = item.split("=", 1)
        run_dirs = [os.path.basename(p) for p in glob.glob(os.path.join("runs", pattern))]
        run_dirs = sorted(run_dirs)
        if not run_dirs:
            raise ValueError(f"No runs matched pattern: {pattern}")
        groups.append((name, run_dirs))

    # Common overlap region across all runs
    mins, maxs = [], []
    for _, rds in groups:
        for rd in rds:
            x, _ = load_curve(rd, args.metric)
            mins.append(float(np.min(x)))
            maxs.append(float(np.max(x)))

    x0 = max(mins)
    x1 = min(maxs)
    if x1 <= x0:
        raise ValueError("Runs do not overlap in timesteps enough to compare.")

    grid = np.linspace(x0, x1, int(args.grid_points))

    plt.figure()

    for name, rds in groups:
        Ys = []
        for rd in rds:
            x, y = load_curve(rd, args.metric)
            y_grid = resample_to_grid(x, y, grid)
            Ys.append(y_grid)

        Y = np.vstack(Ys)
        mean = Y.mean(axis=0)
        se = Y.std(axis=0) / np.sqrt(Y.shape[0])

        # simple moving average smoothing
        w = max(1, int(args.smooth))
        if w > 1:
            kernel = np.ones(w, dtype=float) / float(w)
            mean = np.convolve(mean, kernel, mode="same")
            se = np.convolve(se, kernel, mode="same")

        plt.plot(grid, mean, label=f"{name} (n={len(rds)})")
        plt.fill_between(grid, mean - se, mean + se, alpha=0.2)

    plt.xlabel("timestep")
    plt.ylabel(args.metric)
    plt.title(f"{args.metric}: mean ± SE across seeds")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
