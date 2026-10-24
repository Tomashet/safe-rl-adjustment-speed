# ExpertV5 — Safe Nonstationary Driving Experiments (SB3 + highway-env)

This repository provides a **context-driven nonstationary training setup** for `highway-env`, with:

- **Markov context switching** (nonstationary regimes)
- A **Safety Shield** wrapper (MPC-like + optional conformal calibration)
- A new **Adjustment-Speed Safety Constraint** module that monitors whether the agent can adapt fast enough to the current shift rate and tightens safety when it cannot.

It supports **discrete** agents (DQN / PPO) and **continuous** agents (SAC), with CSV + TensorBoard logging and plotting utilities.

---

## 1) Installation

### 1.1 Create and activate a venv (Windows PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 1.2 Install dependencies

```powershell
pip install -r requirements.txt
```

If you need to install manually:

```powershell
pip install gymnasium highway-env stable-baselines3 torch numpy pandas matplotlib tensorboard
```

---

## 2) Quick Sanity Checks

### 2.1 Imports

```powershell
python -c "import src.adjust_speed; import src.wrappers; import src.safety; print('IMPORTS OK')"
```

### 2.2 Environment creation

```powershell
python -c "from scripts.common import make_env; from src.safety import SafetyParams; e,_,_=make_env('highway-v0',0,'discrete',0.8,False,False,SafetyParams(horizon_n=10,epsilon=0.5)); print('ENV OK'); e.close()"
```

If this fails, fix it before running long training.

---

## 3) Core Concepts

### 3.1 Nonstationarity via Markov context switching

A context scheduler changes the environment regime at episode boundaries. The active context id is propagated via `info["ctx_id"]` each step.

### 3.2 Safety Shield

A wrapper filters actions when safety risk is high (e.g., clearance too small). Optional conformal calibration inflates a safety threshold.

### 3.3 Adjustment-Speed Safety Constraint (NEW)

The module estimates:

- **Shift speed** `S_env`: how fast context/regime changes (or how quickly context embeddings change)
- **Adaptation speed** `S_agent`: a proxy based on policy parameter update magnitude
- `unsafe = (S_env > S_agent + margin)`

When unsafe, the wrapper can pass an `eps_override` to the shield, making it more conservative.

---

## 4) Training Entry Points

### 4.1 Discrete training: DQN / PPO

DQN:

```powershell
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 200000 --seed 0 --run_dir highway_dqn_seed0
```

PPO:

```powershell
python -m scripts.train_discrete --env highway-v0 --algo ppo --total_steps 200000 --seed 0 --run_dir highway_ppo_seed0
```

You can also use `merge-v0`:

```powershell
python -m scripts.train_discrete --env merge-v0 --algo dqn --total_steps 200000 --seed 0 --run_dir merge_dqn_seed0
```

### 4.2 Continuous training: SAC

```powershell
python -m scripts.train_continuous --env highway-v0 --total_steps 300000 --seed 0 --run_dir highway_sac_seed0
```

---

## 5) Adjustment-Speed Monitoring

Enable the adjustment-speed constraint via `--adjust_speed`.

### 5.1 Discrete example

```powershell
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 200000 --seed 0 `
  --adjust_speed --run_dir highway_dqn_adj_seed0
```

### 5.2 Continuous example

```powershell
python -m scripts.train_continuous --env highway-v0 --total_steps 300000 --seed 0 `
  --adjust_speed --run_dir highway_sac_adj_seed0
```

### 5.3 Tuning knobs

- `--adj_shift_window` (default 200): window for shift speed
- `--adj_metric` (`discrete` or `l2`): use discrete ctx switches or embedding deltas
- `--adj_adapt_window` (default 20): number of updates for adaptation speed average
- `--adj_margin` (default 0.0): unsafe if `S_env > S_agent + margin`
- `--adj_temp` (default 10.0): sigmoid temperature for risk score

Example:

```powershell
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 200000 --seed 0 `
  --adjust_speed --adj_margin 0.02 --adj_temp 15 --run_dir tuned_adj_dqn_seed0
```

---

## 6) Safety Shield Controls

Control shield components with:

- `--no_mpc`
- `--no_conformal`

### 6.1 Disable shield entirely (baseline)

```powershell
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 200000 --seed 0 `
  --no_mpc --no_conformal --run_dir baseline_no_shield
```

### 6.2 Enable shield + adjustment speed (full system)

```powershell
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 200000 --seed 0 `
  --adjust_speed --run_dir full_shield_plus_adj
```

---

## 7) Outputs and Logging

Each run writes into:

```
runs/<run_dir>/
  config.json
  train_monitor.csv
  models/final_model.zip
  events.out.tfevents...  (TensorBoard)
```

The CSV contains columns such as:

- `timestep`
- `ctx_id`
- `violation`, `near_miss`
- `shield_used`, `shield_reason`, `eps`, `inflate`
- `adj_risk`, `adj_unsafe`, `adj_s_env`, `adj_s_agent`, `adj_eps_override`

---

## 8) Live Visualization (sanity/debug)

While training is running:

```powershell
python -m scripts.plot_live_sanity --run_dir <run_dir>
```

Optional parameters:

```powershell
python -m scripts.plot_live_sanity --run_dir <run_dir> --window 4000 --pause 1.0 --viol_roll 500
```

This shows 4 panels:
1) `ctx_id`
2) rolling violation rate
3) `adj_risk`
4) `adj_unsafe`

---

## 9) Recommended experiment sequence

Run these in order:

1) Imports + env creation
2) **Baseline** (no shield, no adjust-speed, 5k steps)
3) **Adjust-speed only** (no shield, adjust-speed ON, 8k steps)
4) **Shield only** (shield ON, adjust-speed OFF, 12k steps)
5) **Full system** (shield ON, adjust-speed ON, 20k steps)
6) **Full runs** (200k+ steps, seeds 0/1/2)

Example commands:

```powershell
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 5000  --seed 0 --no_mpc --no_conformal --run_dir T1_baseline
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 8000  --seed 0 --no_mpc --no_conformal --adjust_speed --run_dir T2_adj_only
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 12000 --seed 0 --run_dir T3_shield_only
python -m scripts.train_discrete --env highway-v0 --algo dqn --total_steps 20000 --seed 0 --adjust_speed --run_dir T4_shield_plus_adj
```

---

## 10) Plotting results after training

### 10.1 Plot one run quickly

```powershell
python -m scripts.plot_run --run_dir F_dqn_seed0 --window 2000
```

### 10.2 Compare methods across seeds (paper-grade)

Example patterns:

- baseline: `B_dqn_seed0,1,2`
- adjustment-speed: `F_dqn_seed0,1,2`

```powershell
python -m scripts.plot_compare `
  --groups baseline=B_dqn_seed* adj=F_dqn_seed* `
  --metric violation `
  --smooth 50 `
  --grid_points 2000
```

Also useful:

```powershell
python -m scripts.plot_compare --groups baseline=B_dqn_seed* adj=F_dqn_seed* --metric adj_risk --smooth 50
python -m scripts.plot_compare --groups baseline=B_dqn_seed* adj=F_dqn_seed* --metric shield_used --smooth 50
```

---

## 11) Troubleshooting

### “ModuleNotFoundError: src.adjust_speed”
Ensure:
- `src/__init__.py` exists
- `src/adjust_speed/__init__.py` exists
- you run from repo root

### “ConformalCalibrator got unexpected keyword argument 'params'”
Use the patched `src/safety.py` that supports `ConformalCalibrator(params=SafetyParams(...))`.

### “train_monitor.csv not appearing”
Check:
- you are writing into `runs/<run_dir>/`
- training is actually progressing (timesteps increasing)
- `TrainLoggerCallback` is installed

---

## 12) Research context

This repo supports experiments for research on:
- context-conditioned safety under nonstationarity
- proactive safety via context prediction
- **adjustment speed as a safety constraint** in nonstationary RL
