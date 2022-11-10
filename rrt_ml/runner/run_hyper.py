from rrt_ml.utilities.hyper import *


def run_hyper(cfg):

    hyper = Hyper(cfg)

    if cfg.general.is_train:
        hyper.train()

    else:
        hyper.test()
