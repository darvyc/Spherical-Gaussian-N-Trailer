from sga_ntrailer.config import NTrailerConfig
from sga_ntrailer.path import Path2D
from sga_ntrailer.train_bc import train_and_save


def main() -> None:
    path = Path2D.from_waypoints([(0, 0), (-3, 0.2), (-6, 1.2), (-9, -0.7), (-12, 0.1)])
    config = NTrailerConfig(n_trailers=3, seed=3)
    meta = train_and_save(config, path, out="runs/demo_policy.pt", episodes=4, steps=80, epochs=3, seed=3)
    print(meta)


if __name__ == "__main__":
    main()
