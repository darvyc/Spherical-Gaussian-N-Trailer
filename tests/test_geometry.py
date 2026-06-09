import numpy as np

from sga_ntrailer.geometry import exp_map_s2, log_map_s2, normalize, sphere_point, tangent_coordinates


def test_log_exp_round_trip_small_arc():
    base = sphere_point(0.2, 0.1)
    point = sphere_point(0.35, 0.15)
    tangent = log_map_s2(base, point)
    recon = exp_map_s2(base, tangent)
    assert np.allclose(recon, normalize(point), atol=1e-6)


def test_tangent_coordinates_shape():
    base = sphere_point(0.0, 0.0)
    point = sphere_point(0.1, 0.2)
    coords = tangent_coordinates(base, point)
    assert coords.shape == (2,)
    assert np.all(np.isfinite(coords))
