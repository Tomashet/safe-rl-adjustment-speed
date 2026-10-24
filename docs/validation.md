# Validation and cleanup

## Provenance

The first import commit preserves all 43 files from `Tomashet/ExpertV5-main`. Every imported Git blob hash matched the source. The original repository was not modified. Cleanup is a separate commit in this repository.

## Changes

- Replaced the landing README with a tested setup, implementation map, and explicit scientific scope.
- Preserved the historical README in `docs/legacy-readme.md`.
- Added a compatible demo dependency file without SB3 extras that pull unrelated Atari dependencies.
- Fixed indentation in `scripts/run_all.py`; the batch export now remains inside `main`.
- Removed unsupported `delta_nearmiss`, `d0`, and `h` constructor arguments from Python presets. These were not implemented by `SafetyParams`; removing them does not implement those mechanisms. Historical JSON configurations still need reconciliation before use.
- Made `no_mpc` bypass the distance filter even when calibration is enabled. This corrects the ablation switch; it is a behavior change from the imported prototype.
- Clarified filter comments: the IDLE/zero-control replacement is not verified braking or a safe fallback.

## Checks

Environment: Python 3.12, CPU PyTorch, Gymnasium 0.29.1, highway-env 1.8.2, Stable-Baselines3 2.3.2, NumPy 1.26.4.

`python -m pytest -q` exercises preset construction, the disabled-filter regression, known context-switch frequency, and a 96-step DQN run with actual optimizer updates and checkpoint save/reload.

`python -m compileall -q src scripts tests` checks Python syntax. A 128-step preset CLI run checks logging and checkpoint output. These are execution checks, not evidence of safety improvements, convergence, or paper-result reproduction.
