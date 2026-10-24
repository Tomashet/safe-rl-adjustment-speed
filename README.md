# Adjustment Speed for Nonstationary Reinforcement Learning

Research prototype by Tim Tomashevskiy exploring how runtime constraints can respond when an environment changes faster than an RL agent adapts. The implementation combines an episodic context scheduler, environment-shift and policy-change diagnostics, and a runtime clearance filter in `highway-env`.

This repository presents a compact implementation associated with the research direction **Adjustment Speed as a Safety Constraint for Nonstationary RL**. It is a code demonstration, not a verified reproduction of the paper's experimental results. The CoLLAs context-based constraints paper is separate work; its collision-reduction results should not be attributed to this repository.

## Quick start

Use Python 3.12 from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-demo.txt
python -m pytest -q
python -m scripts.train_discrete --preset highway_discrete_default --total_steps 128 --adjust_speed --run_dir quickstart
```

The 128-step command checks the command-line path and saves a checkpoint; the preset starts DQN learning after 10,000 steps, so this short command does not train the policy. The integration test deliberately lowers that threshold and verifies gradient updates and checkpoint loading.

For a longer exploratory run:

```bash
python -m scripts.train_discrete --preset highway_discrete_default --total_steps 200000 --adjust_speed --run_dir highway_as_seed0
```

Outputs are written under `runs/<run_dir>/`: configuration, training CSV, TensorBoard logs, and `models/final_model.zip`. Run directories are excluded from version control. For headless systems, set `SDL_VIDEODRIVER=dummy`; `OMP_NUM_THREADS=1` is useful for short CPU checks.

## Implementation

| Component | Location | Behavior |
| --- | --- | --- |
| Context schedule | `src/context.py` | Markov transitions among 27 context combinations at episode boundaries |
| Environment wrappers | `src/wrappers.py` | Apply density, reward-speed range, observation noise/dropout, and log collisions |
| Shift estimator | `src/adjust_speed/shift_speed.py` | Rolling context-switch frequency, or embedding-distance proxy |
| Adaptation estimator | `src/adjust_speed/adaptation_speed.py` | Rolling policy-parameter displacement sampled at rollout boundaries |
| Monitor and callback | `src/adjust_speed/` | Compare the proxies and pass a heuristic risk score to the filter |
| Runtime filter | `src/safety.py` | Current-distance threshold increased by the adjustment-risk signal |

The context label `aggr` changes the reward-speed range; it does not implement a separately validated traffic-aggressiveness model. Context IDs are supplied by the simulator, rather than inferred by a learned encoder.

## Interpretation and limits

- Shift frequency and parameter displacement have different units. Their comparison is an experimental heuristic requiring calibration; parameter movement does not establish successful adaptation.
- The filter checks current Euclidean distance. Despite its historical class name, it performs no MPC trajectory optimization or learned forecasting. Its discrete replacement is IDLE (action 1), and its continuous replacement is zero control. Neither is a verified collision-free fallback.
- The calibrator is a rolling empirical quantile helper. The current wrapper does not feed residuals into it, so it provides no meaningful fitted calibration or conformal coverage claim.
- `violation` logs the environment's collision flag. `near_miss` defaults to false and clearance may be unavailable; these columns must not be interpreted as validated measurements. Step collision rates differ from episode collision rates.
- The validated path is single-environment discrete DQN on `highway-v0`. Continuous, batch, plotting, and evaluation scripts are retained for further development. The evaluation script does not replay the training-time adjustment callback and should not be used to claim an AS-on evaluation comparison. Continuous `merge-v0` is unsupported.

See [validation and cleanup notes](docs/validation.md) for checks and changes. Historical instructions are retained in [the archived README](docs/legacy-readme.md), `READ1`, and `INSTRUCTIONS/`; they may describe functionality beyond the current implementation.

## License

MIT; see [LICENSE](LICENSE).
