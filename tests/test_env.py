import numpy as np

from sga_ntrailer.config import NTrailerConfig
from sga_ntrailer.dynamics import NTrailerState
from sga_ntrailer.env import NTrailerReverseEnv
from sga_ntrailer.path import Path2D


def straight_reverse_env() -> NTrailerReverseEnv:
    config = NTrailerConfig(n_trailers=1)
    path = Path2D.from_waypoints([(0.0, 0.0), (-10.0, 0.0)])
    return NTrailerReverseEnv(config=config, path=path, seed=0)


def reward_for_state(env: NTrailerReverseEnv, state: NTrailerState) -> float:
    _, info = env.encoder(state.data, env.path)
    return env.reward(state, steer=0.0, prev_steer=0.0, info=info)


def test_reverse_heading_is_rewarded_over_forward_heading():
    env = straight_reverse_env()
    correct_reverse = NTrailerState.from_values(0.0, 0.0, 0.0, [0.0])
    forward_facing = NTrailerState.from_values(0.0, 0.0, np.pi, [0.0])

    assert reward_for_state(env, correct_reverse) > reward_for_state(env, forward_facing)


def test_progress_weight_adds_dense_progress_bonus():
    env = straight_reverse_env()
    start = NTrailerState.from_values(0.0, 0.0, 0.0, [0.0])
    halfway = NTrailerState.from_values(-5.0, 0.0, 0.0, [0.0])

    assert reward_for_state(env, halfway) > reward_for_state(env, start)


def test_step_reports_reverse_heading_error():
    env = straight_reverse_env()
    env.reset(noise=False)
    _, _, _, info = env.step(0.0)

    assert abs(info["heading_error"]) < 1e-6
