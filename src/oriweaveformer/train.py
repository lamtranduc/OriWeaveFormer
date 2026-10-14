"""Train OriWeaveFormer on one CUDA device or with torchrun DistributedDataParallel."""
import argparse
import json

from .config import config_dict, load_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    for key in ("data-root", "output-dir", "init-checkpoint", "split-manifest"):
        parser.add_argument("--" + key)
    for key in ("epochs", "batch-size", "accumulation-steps", "num-workers", "crop-size"):
        parser.add_argument("--" + key, type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--print-config", action="store_true")
    args = vars(parser.parse_args())
    path, print_config = args.pop("config"), args.pop("print_config")
    cfg = load_experiment(path, args)
    if print_config:
        print(json.dumps(config_dict(cfg), indent=2))
        return
    from .engine import run_training
    run_training(cfg)


if __name__ == "__main__":
    main()
