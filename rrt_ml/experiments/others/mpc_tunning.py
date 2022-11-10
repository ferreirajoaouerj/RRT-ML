from itertools import product

import numpy as np
import pandas as pd
from rich.console import Console

from rrt_ml.environments.car_navigation_bullet_env import *
from rrt_ml.utilities.analytic import *
from rrt_ml.utilities.configs import *


cfg = MasterConfig().copy(deep=True)
cfg.env.general.seed = 0
cfg.env.general.gui = False
cfg.env.general.stats = True
cfg.env.general.max_timestep = 1000
cfg.env.target.pose = 1.5
cfg.env.car.pose = [0, 0, 0]
cfg.env.car.f_val = 240

# Initialize env and reset master
env = CarNavigationBulletEnv(cfg)
env.reset_master()

# Initialize dataframe
df = pd.DataFrame(columns=['nome', 'tempo', 'distancia', 'sucessos'])

# Params to search
l_weight_angle = [0.01, 0.05, 0.20, 0.40]
l_curvature = [2.5, 2.4, 2.3, 2.2]

# Logger
console = Console(width=150)

# it = product(l_weight_angle, l_step_size_mpc, l_n_horizon, l_curvature)]
i = 0
for i, params in enumerate(product(l_weight_angle, l_curvature)):

    # Change config
    cfg.rrt.mpc.weight_angle = params[0]
    cfg.rrt.rs.curvature = params[1]

    # Initialize controller
    rsmpc = MPCRS(cfg)

    # Set name
    nome = f'weight_angle:{params[0]}/curvature:{params[1]}'

    # Initialize sums
    distancia = 0
    tempo = 0
    n_success = 0

    # Log
    console.print(
            f"\n[cyan bold underline]Testing config:[/][cyan]{nome}[/]"
    )

    # Reset to keep episodes equal
    env.reset_master()

    # Get episodes infos
    for ep in range(10):

        # Control loop
        done, obs, info = False, env.reset(), None
        while not done:
            action = rsmpc.get_action(obs)
            obs, reward, done, info = env.step(action)

        # Log
        console.print(
            f"[cyan bold underline]Episode:[/] [cyan]{ep}[/]"
        )

        # Only add info on success
        if info['done_info']['success']:

            # Track num of success to take mean
            n_success += 1

            # Get length and time
            length = env.stats.get_distance_traveled(ep)
            time = env.stats.get_time_to_reach(ep)

            # Add to sum
            distancia += length
            tempo += time

    console.print(
        f"[red bold underline]Done config:[/][red]{i}[/]"
    )

    # Now we take the mean
    if n_success > 0:
        distancia /= n_success
        tempo /= n_success
    else:
        distancia = np.inf
        tempo = -np.inf

    # Add to dataframe
    df = pd.concat([pd.DataFrame([[nome, tempo, distancia, n_success]], columns=df.columns), df], ignore_index=True)

    # Save to file
    df.to_csv('mpc_tuning.csv')

