import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="e.g., F_dqn_seed0")
    ap.add_argument("--window", type=int, default=2000, help="rolling window")
    args = ap.parse_args()

    csv_path = os.path.join("runs", args.run_dir, "train_monitor.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    if "timestep" not in df.columns:
        raise ValueError("Expected a 'timestep' column in train_monitor.csv")

    t = df["timestep"].to_numpy()

    # Rolling violation rate
    viol = df.get("violation", 0).astype(float)
    viol_roll = viol.rolling(args.window, min_periods=1).mean()

    plt.figure()
    plt.plot(t, viol_roll)
    plt.xlabel("timestep")
    plt.ylabel(f"violation rate (roll={args.window})")
    plt.title(f"{args.run_dir}: Violation rate")
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.show()

    # Adjustment-speed risk
    if "adj_risk" in df.columns:
        plt.figure()
        plt.plot(t, df["adj_risk"].astype(float).to_numpy())
        plt.xlabel("timestep")
        plt.ylabel("adj_risk")
        plt.title(f"{args.run_dir}: Adjustment-speed risk")
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.show()

    # Shield usage
    if "shield_used" in df.columns:
        shield = df["shield_used"].astype(float)
        shield_roll = shield.rolling(args.window, min_periods=1).mean()
        plt.figure()
        plt.plot(t, shield_roll)
        plt.xlabel("timestep")
        plt.ylabel(f"shield usage rate (roll={args.window})")
        plt.title(f"{args.run_dir}: Shield usage")
        plt.ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
