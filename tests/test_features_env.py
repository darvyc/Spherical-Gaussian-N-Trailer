import numpy as np

from sga_ntrailer.config import NTrailerConfig
from sga_ntrailer.env import NTrailerReverseEnv
from sga_ntrailer.features import GaussianCombinatorialFeatures, SphericalGaussianEncoder
from sga_ntrailer.path import Path2D


def test_encoder_output_dimension():
    config = NTrailerConfig(n_trailers=2)
    path = Path2D.from_waypoints([(0, 0), (-2, 0.2), (-4, 0.0)])
    enc = SphericalGaussianEncoder(config, rff_dim=8, subset_dim=5, seed=1)
    state = np.array([0, 0, np.pi, 0.0, 0.0], dtype=float)
    x, info = enc(state, path)
    assert x.shape == (enc.out_dim,)
    assert np.all(np.isfinite(x))
    assert np.isfinite(info.cross_track)


def test_env_step():
    config = NTrailerConfig(n_trailers=2)
    path = Path2D.from_waypoints([(0, 0), (-3, 0.0)])
    env = NTrailerReverseEnv(config, path, seed=1)
    obs = env.reset(noise=False)
    obs2, reward, done, info = env.step([0.0])
    assert obs.shape == obs2.shape
    assert np.isfinite(reward)
    assert isinstance(done, bool)
    assert "cross_track" in info


def test_gaussian_features_can_disable_random_blocks():
    features = GaussianCombinatorialFeatures(in_dim=3, rff_dim=0, subset_dim=0, seed=1)
    x = features(np.array([0.2, -0.1, 0.4], dtype=float))

    assert x.shape == (3,)
    assert np.allclose(x, [0.2, -0.1, 0.4])
