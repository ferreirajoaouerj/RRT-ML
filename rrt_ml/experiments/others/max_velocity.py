import numpy as np
import pybullet as p
from matplotlib import pyplot as plt
from scipy import interpolate

from rrt_ml.environments.car_navigation_bullet_env import *
from rrt_ml.utilities.configs import *


# Set stats to true to track bullet stats
cfg = MasterConfig()
cfg.env.general.stats = True
cfg.env.general.gui = True
cfg.env.general.max_timestep = 480

# Reset master to track stats
env = CarNavigationBulletEnv(cfg)
env.reset_master()

# Generate info
env.reset()
done = False
while not done:
    obs, reward, done, info = env.step([1, 1])

# Get variables
t = env.stats.bullet.time[0, :50, 0]
v = env.stats.bullet.car_velocities[0, :51, 0]
a = np.diff(v)
v = v[:50]

# Interpolate
fa = interpolate.interp1d(t, a, kind='zero')
fv = interpolate.interp1d(t, v, kind='zero')
x = np.linspace(t[0], t[-1], 10000)
a_x = fa(x)
v_x = fv(x)

# Config plot
plt.rcParams['axes.grid'] = True

# Plot
fig, axs = plt.subplots(1, 2, squeeze=True)
axs[0].plot(x, v_x, label='Velocidade (m/s)')
axs[0].set_xlabel('Tempo (s)')
axs[0].legend()
axs[1].plot(x, a_x, label='Aceleração (m/s^2)', color='red')
axs[1].set_xlabel('Tempo (s)')
axs[1].legend()
fig.savefig('max_velocity.png', dpi=300)
pass

