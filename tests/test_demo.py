from types import SimpleNamespace

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import DQN
from scripts.common import make_env
from scripts.presets import PRESETS
from src.safety import SafetyParams, MPCLikeSafetyShield
from src.adjust_speed import (ShiftSpeedConfig, ShiftSpeedEstimator,
    AdaptSpeedConfig, AdaptationSpeedEstimator, FeasibilityConfig,
    FeasibilityMonitor, AdjustSpeedSafetyCallback)


def test_presets_match_implemented_parameters():
    for preset in PRESETS.values():
        SafetyParams(**preset['safety'])


def test_disabled_filter_preserves_action_near_vehicle():
    ego = SimpleNamespace(position=np.array([0., 0.]))
    other = SimpleNamespace(position=np.array([1., 0.]))
    env = SimpleNamespace(unwrapped=SimpleNamespace(
        road=SimpleNamespace(vehicles=[ego, other]), vehicle=ego))
    shield = MPCLikeSafetyShield(SafetyParams(), no_mpc=True)
    action, meta = shield.filter_action(env, 3, 0)
    assert action == 3 and not meta['shield_used']
    shield.no_mpc = False
    action, meta = shield.filter_action(env, 3, 0)
    assert action == 1 and meta['shield_used']


def test_shift_frequency():
    shift = ShiftSpeedEstimator(ShiftSpeedConfig(window=10))
    for context in [0, 0, 1, 1, 2]:
        shift.update(context)
    assert shift.speed() == 0.5


def test_training_updates_and_checkpoint_reload(tmp_path):
    env, _, _ = make_env('highway-v0', 0, 'discrete', 0.2,
                         False, False, SafetyParams())
    env = TimeLimit(env, max_episode_steps=12)
    adapt = AdaptationSpeedEstimator(AdaptSpeedConfig(window_updates=10))
    callback = AdjustSpeedSafetyCallback(
        ShiftSpeedEstimator(ShiftSpeedConfig(window=50)), adapt,
        FeasibilityMonitor(FeasibilityConfig()))
    try:
        model = DQN('MlpPolicy', env, learning_starts=16, buffer_size=1000,
                    batch_size=16, train_freq=4, seed=0, verbose=0)
        model.learn(96, callback=callback)
        assert model._n_updates > 0
        assert adapt.speed() > 0
        path = tmp_path / 'model'
        model.save(path)
        restored = DQN.load(path)
        obs, _ = env.reset(seed=1)
        action, _ = restored.predict(obs, deterministic=True)
        _, _, _, _, info = env.step(action)
        assert np.isfinite(info['adj_risk'])
        assert 'violation' in info and 'ctx_id' in info
    finally:
        env.close()


def test_json_configs_match_presets():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    configs = {p.stem: json.loads(p.read_text()) for p in (root / 'configs').glob('*.json')}
    assert configs == PRESETS


def test_continuous_merge_rejected():
    import pytest
    with pytest.raises(ValueError, match='discrete actions only'):
        make_env('merge-v0', 0, 'continuous', 0.8, False, False, SafetyParams())
