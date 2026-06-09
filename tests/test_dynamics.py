import numpy as np

from sga_ntrailer.config import NTrailerConfig
from sga_ntrailer.dynamics import NTrailerState, is_jackknifed, step_kinematics, unit_poses


def test_step_has_finite_state():
    config = NTrailerConfig(n_trailers=3)
    state = NTrailerState.from_values(0, 0, np.pi, [0.0, 0.0, 0.0])
    nxt = step_kinematics(state, steer=0.1, config=config)
    assert nxt.data.shape == (6,)
    assert np.all(np.isfinite(nxt.data))


def test_unit_poses_count():
    config = NTrailerConfig(n_trailers=4)
    state = NTrailerState.from_values(0, 0, 0, [0.0, 0.1, -0.1, 0.0])
    poses = unit_poses(state, config)
    assert poses.shape == (5, 3)


def test_jackknife_detection():
    config = NTrailerConfig(n_trailers=1)
    state = NTrailerState.from_values(0, 0, 0, [config.max_articulation + 0.01])
    assert is_jackknifed(state, config)
