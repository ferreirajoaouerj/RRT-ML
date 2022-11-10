import numpy as np

from rrt_ml.environments.car_navigation_bullet_env import *
from rrt_ml.utilities.configs import *


cfg = MasterConfig()
cfg.env.general.gui = False
cfg.env.car.f_val = 240
cfg.env.general.max_timestep = 2000
env = CarNavigationBulletEnv(cfg)

done = False
obs = env.reset()
i = 0
while not done:
    obs, reward, done, info = env.step(env.action_space.sample())


