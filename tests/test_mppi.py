import numpy as np
import pytest

from sga_ntrailer.config import NTrailerConfig
from sga_ntrailer.env import NTrailerReverseEnv
from sga_ntrailer.mppi import MPPIController
from sga_ntrailer.path import Path2D


def test_mppi_action_bounded():
    config = NTrailerConfig(n_trailers=1)
    path = Path2D.from_waypoints([(0, 0), (-2, 0.1), (-4, 0.0)])
    env = NTrailerReverseEnv(config, path, seed=4)
    obs = env.reset(noise=False)
    ctrl = MPPIController(env, horizon=3, samples=5, seed=4)
    action = ctrl.act(obs)
    assert action.shape == (1,)
    assert np.abs(action[0]) <= config.max_steer + 1e-9


def test_mppi_zero_noise_is_deterministic():
    config = NTrailerConfig(n_trailers=1)
    path = Path2D.from_waypoints([(0, 0), (-2, 0.1), (-4, 0.0)])
    env = NTrailerReverseEnv(config, path, seed=4)
    obs = env.reset(noise=False)
    ctrl = MPPIController(env, horizon=3, samples=5, noise_std=0.0, seed=4)

    action = ctrl.act(obs)

    assert ctrl.noise_std == 0.0
    assert ctrl.last_diag.action_std == 0.0
    assert action[0] == 0.0


def test_mppi_rejects_empty_sampling_grid():
    config = NTrailerConfig(n_trailers=1)
    path = Path2D.from_waypoints([(0, 0), (-2, 0.1), (-4, 0.0)])
    env = NTrailerReverseEnv(config, path, seed=4)

    with pytest.raises(ValueError, match="horizon"):
        MPPIController(env, horizon=0)
    with pytest.raises(ValueError, match="samples"):
        MPPIController(env, samples=0)
