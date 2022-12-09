import sys
from argparse import ArgumentParser
from pathlib import Path

# Add dependencies
sys.path.append(str(Path(__file__).parent / 'deps/libraries'))

from rrt_ml.runner.run_rl import *
from rrt_ml.runner.run_sl import *
from rrt_ml.runner.run_rrt import *
from rrt_ml.runner.run_hyper import *
from rrt_ml.utilities.configs import MasterConfig


def run_rrt_ml():

    # Parse args
    parser = ArgumentParser()
    group_algo = parser.add_mutually_exclusive_group(required=True)
    group_algo.add_argument("--rl", action="store_true")
    group_algo.add_argument("--sl", action="store_true")
    group_algo.add_argument("--rrt", action="store_true")
    group_train = parser.add_mutually_exclusive_group(required=True)
    group_train.add_argument("--train", action="store_true")
    group_train.add_argument("--test", action="store_true")
    parser.add_argument("--hyper", action="store_true", required=False, default=False)
    parser.add_argument("--config", type=str, required=False)
    args = parser.parse_args()

    # Load config
    cfg = MasterConfig.load(args)

    if not cfg.general.is_hyper:
        match cfg.general.algorithm:
            case 'rl':
                run_rl(cfg)
            case 'rrt':
                run_rrt(cfg)
            case 'sl':
                run_sl(cfg)
            case _:
                raise NotImplementedError

    else:
        run_hyper(cfg)


if __name__ == '__main__':
    run_rrt_ml()
