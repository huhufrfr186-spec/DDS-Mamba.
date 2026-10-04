"""Sequential full-model retraining under S19 seeds; no test-set tuning."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--dev-manifest", required=True)
    parser.add_argument("--assets", default="assets")
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--out", default="runs/three_seeds")
    parser.add_argument("--backend", default="reference", choices=["reference", "mamba_ssm"])
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    for seed in (2024, 2025, 2026):
        command = [sys.executable, "-m", "dds_mamba", "train", "--seed", str(seed),
                   "--train-manifest", str(Path(args.train_manifest).resolve()),
                   "--dev-manifest", str(Path(args.dev_manifest).resolve()),
                   "--assets", str(Path(args.assets).resolve()), "--config", str(Path(args.config).resolve()),
                   "--out", str(Path(args.out).resolve() / f"seed_{seed}"), "--backend", args.backend, "--device", args.device]
        subprocess.run(command, cwd=project, check=True)


if __name__ == "__main__":
    main()
