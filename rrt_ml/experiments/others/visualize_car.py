from rrt_ml.environments.car_navigation_bullet_env import *
from rrt_ml.utilities.configs import *


cfg = MasterConfig()
cfg.env.general.gui = True

env = CarNavigationBulletEnv(cfg)

env.reset()
