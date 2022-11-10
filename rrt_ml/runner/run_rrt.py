from rrt_ml.algorithms.rrt import *
from rrt_ml.utilities.configs import *


def run_rrt(cfg: MasterConfig):

    # Initialize RRT
    rrt = RRT(cfg)

    # Train
    if cfg.general.is_train:
        rrt.train()

    # Test
    else:
        rrt.test()
