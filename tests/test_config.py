import pytest

from sga_ntrailer.config import NTrailerConfig


def test_config_rejects_nonpositive_physical_scales():
    with pytest.raises(ValueError, match="dt"):
        NTrailerConfig(dt=0.0)
    with pytest.raises(ValueError, match="tractor_length"):
        NTrailerConfig(tractor_length=-1.0)
    with pytest.raises(ValueError, match="cross_track_scale"):
        NTrailerConfig(cross_track_scale=0.0)


def test_config_rejects_inconsistent_articulation_limits():
    with pytest.raises(ValueError, match="soft_articulation"):
        NTrailerConfig(soft_articulation=2.0, max_articulation=1.0)


def test_config_rejects_incomplete_reward_weights():
    with pytest.raises(ValueError, match="reward_weights missing"):
        NTrailerConfig(reward_weights={"cross_track": 1.0})
