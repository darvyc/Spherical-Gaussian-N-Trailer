from sga_ntrailer.config import NTrailerConfig
from sga_ntrailer.env import NTrailerReverseEnv
from sga_ntrailer.mppi import MPPIController, rollout_mppi
from sga_ntrailer.path import Path2D
from sga_ntrailer.utils import write_json


def main() -> None:
    path = Path2D.from_waypoints([(0, 0), (-3, 0.2), (-6, 1.2), (-9, -0.7), (-12, 0.1)])
    config = NTrailerConfig(n_trailers=4, seed=42)
    env = NTrailerReverseEnv(config=config, path=path, seed=42)
    controller = MPPIController(env, horizon=20, samples=256, seed=42)
    trace = rollout_mppi(env, controller, steps=220)
    write_json("runs/demo_mppi.json", {"config": {"n_trailers": 4}, "path": path.to_json_dict(), "trace": trace})
    print("wrote runs/demo_mppi.json")


if __name__ == "__main__":
    main()
