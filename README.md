# Spherical Gaussian N-Trailer Reversal

A research-grade Python implementation for learning reverse path following with an articulated vehicle and an arbitrary number of passive trailers.

The project combines three layers:

1. **Kinematic N-trailer simulator** — a standard on-axle nonholonomic chain with configurable vehicle length, trailer lengths, steering limits, speed limits, and jackknife constraints.
2. **Sphere-tangent geometric encoder** — local path geometry is lifted onto `S²`; heading/curvature discrepancies are converted into tangent-space arc coordinates so that a policy sees smooth geometric error signals rather than brittle Cartesian coordinates.
3. **Gaussian combinatorial feature field** — random Fourier features approximate Gaussian RBF structure; sparse signed subset interactions create high-dimensional “string” features that couple tangent directions, path curvature, and articulation angles.

The default control stack uses MPPI as an expert and trains a neural policy by behaviour cloning. This gives a reproducible simulation-to-learning pipeline rather than a hand-tuned steering rule.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[train,plot]

python -m sga_ntrailer.cli simulate --trailers 3 --steps 250 --out runs/demo.json
python -m sga_ntrailer.cli train-bc --trailers 3 --episodes 16 --epochs 8 --out runs/policy.pt
python -m sga_ntrailer.cli eval-policy --trailers 3 --policy runs/policy.pt --steps 250 --out runs/eval.json
python examples/demo_mppi.py
```

For development checks:

```bash
pip install -e .[dev]
python -m pytest
python -m ruff check .
```

## Repository layout

```text
sga_ntrailer/
  dynamics.py          N-trailer kinematics and geometry
  path.py              Arc-length path representation
  geometry.py          S² exponential/log maps and tangent bases
  features.py          Sphere-tangent + Gaussian combinatorial encoder
  env.py               Gym-like reversal environment
  mppi.py              Sampling-based expert controller
  policy.py            PyTorch behaviour-cloned policy
  train_bc.py          Expert dataset generation and policy training
  cli.py               Command-line interface
examples/
  demo_mppi.py         MPPI rollout example
  demo_train.py        Small behaviour-cloning run
  path_s_curve.json    Example S-curve path
docs/
  mathematical_notes.md
  papers.md
  limitations.md
tests/
  pytest coverage for config validation, dynamics, geometry, features, MPPI, and environment
```

## Control formulation

Let the state be

```text
z = [x0, y0, theta0, alpha1, ..., alphaN]
```

where `(x0, y0, theta0)` is the tractor pose and `alpha_i = theta_{i-1} - theta_i` is the articulation angle between unit `i-1` and unit `i`. The simulator recursively propagates trailer yaw rates using the standard kinematic chain

```text
v_i = v_{i-1} cos(alpha_i)
theta_dot_i = v_i sin(alpha_i) / L_i
alpha_dot_i = theta_dot_{i-1} - theta_dot_i
```

with negative longitudinal speed for reversing and bounded steering input.

For a local path tangent heading `psi` and curvature proxy `kappa`, the encoder lifts path geometry to a unit sphere:

```text
q_path = normalize([cos(psi), sin(psi), tanh(kappa / kappa_scale)])
q_vehicle = normalize([cos(theta0), sin(theta0), tanh(e_cross / e_scale)])
```

The logarithmic map `log_{q_path}(q_vehicle)` gives a two-coordinate tangent error. This is fused with articulation angles and RBF-style random Fourier features, then passed to either MPPI or the learned policy.

## What runs out of the box

- MPPI expert on arbitrary `N` within normal CPU limits.
- Behaviour-cloning training from MPPI demonstrations.
- Neural policy evaluation.
- JSON rollout traces with state, action, reward, cross-track error, heading error, and jackknife flags.
- Pure NumPy simulator; PyTorch is only required for training/evaluating the learned policy.

## Validation contracts

The public constructors reject invalid physical scales, articulation limits, reward-weight tables, feature dimensions, and empty MPPI sampling grids. This keeps failures close to the caller instead of surfacing later as divide-by-zero, empty-stack, or controller-weight errors.

The default test suite checks finite RK4 state propagation, jackknife detection, spherical log/exp round trips, encoder output shape, deterministic zero-noise MPPI behavior, bounded MPPI actions, and gym-like environment stepping.

## Research basis

The vehicle model and control problem are grounded in the N-trailer/nonholonomic-control literature, especially differential-flatness treatments of trailer systems and reversing path-following controllers. The sampling expert follows the MPPI family of model predictive controllers. The feature map follows Gaussian-kernel approximation via random Fourier features and manifold-aware Gaussian vector-field ideas. See [`docs/papers.md`](docs/papers.md).

## Minimal Python example

```python
from sga_ntrailer.config import NTrailerConfig
from sga_ntrailer.path import Path2D
from sga_ntrailer.env import NTrailerReverseEnv
from sga_ntrailer.mppi import MPPIController

path = Path2D.from_waypoints([(0, 0), (-4, 0.5), (-8, -0.5), (-12, 0)])
config = NTrailerConfig(n_trailers=3)
env = NTrailerReverseEnv(config=config, path=path, seed=7)
controller = MPPIController(env, horizon=24, samples=256, seed=7)

obs = env.reset()
for _ in range(200):
    action = controller.act(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
```

## Licence

MIT.
