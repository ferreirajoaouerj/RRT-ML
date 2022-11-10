import pickle

import matplotlib.pyplot as plt
import numpy as np
from gekko import GEKKO
from scipy import signal

from rrt_ml.environments.car_navigation_bullet_env import *
from rrt_ml.utilities.configs import *
from rrt_ml.utilities.misc import *


cfg = MasterConfig()
cfg.env.general.stats = True
cfg.env.general.gui = True
cfg.env.general.max_timestep = 2000
cfg.env.car.f_val = 240
cfg.env.car.pose = [0, 0, 0]
env = CarNavigationBulletEnv(cfg)
env.reset_master()
obs = env.reset()

done = False
u = []
while not done:
    action = env.action_space.sample()
    for _ in range(np.random.randint(20, 80, 1)[0]):
        obs, reward, done, info = env.step(action)
        if done:
            break

# Get curves
u_v = remove_trailing_zeros(env.stats.mdp.actions[0, :, 0])
u_phi = remove_trailing_zeros(env.stats.mdp.actions[0, :, 1])
time = remove_trailing_zeros(env.stats.bullet.time[0, :, 0])
phi1 = remove_trailing_zeros(env.stats.bullet.car_steers[0, :, 0])
phi2 = remove_trailing_zeros(env.stats.bullet.car_steers[0, :, 1])
vs = remove_trailing_zeros(env.stats.bullet.car_velocities[0, :, 0])

m = GEKKO()
dt = 1/240
m.time = np.arange(0, 2000*dt-dt, dt)

# Parameters
v_meas = m.Param(value=vs)
uv_meas = m.Param(value=u_v)
v_pred = m.Var()
kvx = m.FV(value=1)
# kphix = m.FV(value=1)
kvu = m.FV(value=1)
# kphiu = m.FV(value=1)

# Available to optimize
kvx.STATUS = 1
# kphix.STATUS = 1
kvu.STATUS = 1
# kphiu.STATUS = 1

# ODE's
m.Equation(v_pred.dt() == kvx*v_pred + kvu*uv_meas)

# Objective
m.Minimize((v_pred-v_meas)**2)

# Application options
m.options.IMODE = 2   # Dynamic Simultaneous - estimation

# Solve
m.solve(disp=True)

# show final objective
print('Final SSE Objective: ' + str(m.options.objfcnval))

# Test
A = [kvx.value[0]]
B = [kvu.value[0]]
C = [1.0]
D = [0.0]
sys = signal.StateSpace(A, B, C, D)
t_model, y_model = signal.step(sys)

# Env
env.reset_master()
env.reset()
for _ in range(25):
    obs, reward, done, info = env.step([1, 0])

vs = remove_trailing_zeros(env.stats.bullet.car_velocities[0, :, 0])

fig, ax = plt.subplots()
ax.plot(np.arange(0, (1/240)*25, (1/240)), vs)
ax.plot(t_model, y_model, '--')
plt.show()
