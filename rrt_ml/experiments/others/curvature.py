from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

from rrt_ml.environments.car_navigation_bullet_env import *
from rrt_ml.utilities.analytic import *
from rrt_ml.utilities.configs import *


cfg = MasterConfig()
cfg.env.general.seed = 0
cfg.env.general.gui = True
cfg.env.general.stats = True
cfg.env.general.max_timestep = 1000
cfg.env.target.pose = 1.5
cfg.env.car.pose = [0, 0, 0]
cfg.env.car.f_val = 240

cfg.env.car.pose = [0, 0, 0]
cfg.env.target.pose = [0, 0.79, np.pi]

cfg.rrt.rs.curvature = 2.45

# Initialize env and reset master
env = CarNavigationBulletEnv(cfg)
env.reset_master()
env.reset()

rsmpc = MPCRS(cfg)
action = rsmpc._set_rs_path(env._get_observation())

fig, ax = plt.subplots()
ax.plot(rsmpc.rs_xs, rsmpc.rs_ys, linewidth=2)
for ep in range(50):

    # Control loop
    done, obs, info = False, env.reset(), None
    [env.step([0, 1]) for _ in range(50)]
    xs, ys = [], []
    while not done:
        obs, reward, done, info = env.step([1, 1])
        xs.append(obs['observation'][0])
        ys.append(obs['observation'][1])
    ax.plot(xs, ys, 'r--')
    fig.show()
