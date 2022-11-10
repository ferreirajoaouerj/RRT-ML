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
cfg.env.car.f_val = 240

# Reset master to track stats
env = CarNavigationBulletEnv(cfg)
env.reset_master()

# Generate info
env.reset()
done = False
vel_rear_wheel = []
pos_front_wheel = []
while not done:
    _, vel, _, _ = p.getJointState(env.car.id, 3, physicsClientId=env.client)
    pos, _, _, _ = p.getJointState(env.car.id, 6, physicsClientId=env.client)
    vel_rear_wheel.append(vel)
    pos_front_wheel.append(pos)
    obs, reward, done, info = env.step([1, 1])

# Get variables
t = env.stats.bullet.time[0, :50, 0]
s = pos_front_wheel[:50]
v = vel_rear_wheel[:50]

# Interpolate
fs = interpolate.interp1d(t, s, kind='zero')
fv = interpolate.interp1d(t, v, kind='zero')
x = np.linspace(t[0], t[-1], 10000)
s_x = fs(x)
v_x = fv(x)

# Config plot
plt.rcParams['axes.grid'] = True

# Plot
fig, axs = plt.subplots(1, 2, squeeze=True)
axs[0].plot(x, v_x, label='Velocidade (rad/s)')
axs[0].set_xlabel('Tempo (s)')
axs[0].legend()
axs[1].plot(x, s_x, label='Posição (rad)', color='red')
axs[1].set_xlabel('Tempo (s)')
axs[1].legend()
fig.savefig('motor_graphs.png', dpi=300)

fig, ax = plt.subplots()
ax.plot(x, v_x, label='Velocidade (rad/s)')
ax.set_xlabel('Tempo (s)')
ax.legend()
fig.savefig("graficosMotor1.pdf", format="pdf", bbox_inches="tight")
fig.savefig("graficosMotor1.png", dpi=300, bbox_inches="tight")

fig, ax = plt.subplots()
ax.plot(x, s_x, label='Posição (rad)', color='red')
ax.set_xlabel('Tempo (s)')
ax.legend()
fig.savefig("graficosMotor2.pdf", format="pdf", bbox_inches="tight")
fig.savefig("graficosMotor2.png", dpi=300, bbox_inches="tight")
pass

