import copy

import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, minimize

from rrt_ml.environments.car_navigation_bullet_env import *
from rrt_ml.utilities.configs import *
from rrt_ml.utilities.formulas import *
from rrt_ml.utilities.misc import *

cfg = MasterConfig()
cfg.env.general.stats = True
cfg.env.general.gui = True
cfg.env.general.max_timestep = 200
cfg.env.car.f_val = 240
cfg.env.car.pose = [0, 0, 0]
env = CarNavigationBulletEnv(cfg)
env.reset_master()
obs = env.reset()

done = False
u = []
while not done:
    if env.timestep < 100:
        u.append(1)
        obs, reward, done, info = env.step([1, 0])
    else:
        u.append(-1)
        obs, reward, done, info = env.step([-1, 0])

# Get curves
time = remove_trailing_zeros(env.stats.bullet.time[0, :, 0])
vs = remove_trailing_zeros(env.stats.bullet.car_velocities[0, :, 0])

# State space model
k = 1.02
tau = 0.05
A = -1.0/tau
B = k / tau
C = 1.0
D = 0.0
sys = signal.StateSpace(A, B, C, D)

tout, yout, xout = signal.lsim(sys, u, time)

fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=True)
axs.scatter(time, vs)
axs.plot(time, yout, color='red')
fig.show()

env.reset_master()
obs = env.reset()
done = False
u = []
while not done:
    if env.timestep < 100:
        u.append(0.7)
        obs, reward, done, info = env.step([0, 1])
    else:
        u.append(-0.7)
        obs, reward, done, info = env.step([0, -1])

time = remove_trailing_zeros(env.stats.bullet.time[0, :, 0])
phi1 = remove_trailing_zeros(env.stats.bullet.car_steers[0, :, 0])
phi2 = remove_trailing_zeros(env.stats.bullet.car_steers[0, :, 1])

vrl, vrr, vfl, vfr, phil, phir = get_ackermann_v_rf_lr_phi_lr(v_ref=0, phi_ref=cfg.env.car.phi_max, cfg=cfg)

a = cfg.env.car.axis_dist
b = cfg.env.car.wheel_dist

phi_ref = np.arctan((2*a*np.tan(phir))/(2*a-b*np.tan(phir)))

phi_refs = []
for p in phi2:
    phi_ref = np.arctan((2 * a * np.tan(p)) / (2 * a - b * np.tan(p)))
    phi_refs.append(phi_ref)

# State space model
k = 1.0
tau = 0.05
A = -1.0/tau
B = k / tau
C = 1.0
D = 0.0
sys = signal.StateSpace(A, B, C, D)

tout, yout, xout = signal.lsim(sys, u, time)

fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=True)
axs.scatter(time, phi1)
axs.scatter(time, phi2)
axs.scatter(time, phi_refs)
axs.plot(time, yout, color='red')
fig.show()

# Second order approximation
# env.reset_master()
# obs = env.reset()
# done = False
# u = []
# for _ in range(100):
#     if env.timestep < 150:
#         u.append(1)
#         obs, reward, done, info = env.step([1, 0])
#     else:
#         u.append(-1)
#         obs, reward, done, info = env.step([-1, 0])
#
# # Get curves
# time = remove_trailing_zeros(env.stats.bullet.time[0, :, 0])
# vs = remove_trailing_zeros(env.stats.bullet.car_velocities[0, :, 0])
#
# # State space model
# k = 1.02
# tau1 = 0.02559706
# tau2 = 0.02559741
# A = [[-1/tau1, 0], [1/tau2, -1/tau2]]
# B = [[k/tau1], [0]]
# C = [0, 1]
# D = 0
# sys = signal.StateSpace(A, B, C, D)
# tout, yout, xout = signal.lsim(sys, u, time)
#
# fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=True)
# axs.scatter(time, vs)
# axs.plot(time, yout, color='red')
# fig.show()

def evaluate(x):
    k = x[0]
    tau1 = x[1]
    tau2 = x[2]
    A = [[-1 / tau1, 0], [1 / tau2, -1 / tau2]]
    B = [[k / tau1], [0]]
    C = [0, 1]
    D = 0
    sys = signal.StateSpace(A, B, C, D)
    tout, yout, xout = signal.lsim(sys, u, time)
    return np.mean(np.power(yout-vs, 2))

x0 = np.array([1.02, 0.1, 0.01])
res = minimize(evaluate, x0, method='BFGS')

pass
















# State space model
k = 1.0
tau = 0.05
A = -1.0/tau
B = k / tau
C = 1.0
D = 0.0
sys = signal.StateSpace(A, B, C, D)
tout, yout, xout = signal.lsim(sys, u, time)

def evaluate(x):
    k = x[0]
    tau = x[1]
    A = -1.0 / tau
    B = k / tau
    C = 1.0
    D = 0.0
    sys = signal.StateSpace(A, B, C, D)
    tout, yout, xout = signal.lsim(sys, u, time)
    return np.mean(np.power(yout-vs, 2))

x0 = np.array([1.02, 0.1, 0.01])
res = minimize(evaluate, x0, method='BFGS')

k = res.x[0]
tau = res.x[1]
A = -1.0 / tau
B = k / tau
C = 1.0
D = 0.0
sys = signal.StateSpace(A, B, C, D)
tout, yout, xout = signal.lsim(sys, u, time)

fig, axs = plt.subplots(nrows=1, ncols=1, squeeze=True)
axs.scatter(time, vs)
axs.plot(time, yout, color='red')
fig.show()