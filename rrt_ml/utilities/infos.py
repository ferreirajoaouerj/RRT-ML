from typing import Any

import numpy as np

from rrt_ml.utilities.formulas import *
from rrt_ml.utilities.hints import *
from rrt_ml.utilities.stats import *


class RRTEpochInfo(BaseStats):
    train_losses: list[float, ...] | None = None
    val_losses: list[float, ...] | None = None

    @classmethod
    def new(cls, train_losses, val_losses):
        epoch_info = cls()
        epoch_info.train_losses = train_losses
        epoch_info.val_losses = val_losses
        return epoch_info


class SLEpochInfo(BaseStats):
    epoch_n: int | None = None
    train_loss: Vector | None = None
    val_loss: Vector | None = None
    train_kl_loss: Vector | None = None
    val_kl_loss: Vector | None = None
    train_recon_loss: Vector | None = None
    val_recon_loss: Vector | None = None

    @classmethod
    def new(cls, epoch_n, t_l, v_l, t_kl_l, v_kl_l, t_r_l, v_r_l):
        self = cls()
        self.epoch_n = epoch_n
        self.train_loss = t_l
        self.val_loss = v_l
        self.train_kl_loss = t_kl_l
        self.val_kl_loss = v_kl_l
        self.train_recon_loss = t_r_l
        self.val_recon_loss = v_r_l
        return self


class RLEpochInfo(BaseStats):
    det_val_reward: float | None = None
    rand_val_reward: float | None = None
    env_stats: EnvStats | None = None
    train_timestep_num: int | None = None

    @classmethod
    def new(cls, det_val_reward, rand_val_reward, env_stats, train_timestep_num):
        self = cls()
        self.det_val_reward = det_val_reward
        self.rand_val_reward = rand_val_reward
        self.env_stats = env_stats
        self.train_timestep_num = train_timestep_num
        return self


class NodeReachInfo(BaseStats):
    cost: None | float | int = None
    done_info: None | dict = None
    length: None | float = None
    rewards: None | np.ndarray = None
    steers1: None | np.ndarray = None
    steers2: None | np.ndarray = None
    timesteps: None | np.ndarray = None
    vs: None | np.ndarray = None
    xs: None | np.ndarray = None
    yaws: None | np.ndarray = None
    ys: None | np.ndarray = None

    @classmethod
    def new_from_bullet(cls, timesteps, done_info, env_stats: EnvStats):

        # Instantiate and assign args
        stats = cls()
        stats.done_info = done_info
        stats.timesteps = timesteps

        # Secondary stats
        stats.rewards = np.trim_zeros(env_stats.bullet.rewards[0, :, 0])
        stats.steers1 = np.trim_zeros(env_stats.bullet.car_steers[0, :, 0])
        stats.steers2 = np.trim_zeros(env_stats.bullet.car_steers[0, :, 1])
        stats.vs = np.trim_zeros(env_stats.bullet.car_velocities[0, :, 0])
        stats.xs = np.trim_zeros(env_stats.bullet.car_poses[0, :, 0])
        stats.yaws = np.trim_zeros(env_stats.bullet.car_poses[0, :, 2])
        stats.ys = np.trim_zeros(env_stats.bullet.car_poses[0, :, 1])

        # Length
        stats.length = path_length(stats.xs, stats.ys)

        # Get cost
        stats.cost = stats.timesteps

        return stats

    @classmethod
    def new_from_rs(cls, timesteps, done_info, poses, rewards):

        # Instantiate and assign args
        stats = cls()
        stats.timesteps = timesteps
        stats.done_info = done_info
        poses = np.array(poses)
        stats.xs = np.array(poses[:, 0])
        stats.yaws = np.array(poses[:, 2])
        stats.ys = np.array(poses[:, 1])
        stats.rewards = np.array(rewards)

        # Secondary stats
        stats.steers1 = None
        stats.steers2 = None
        stats.vs = None

        # Length
        stats.length = path_length(stats.xs, stats.ys)

        # Get cost
        stats.cost = stats.length

        return stats


class RRTSolutionInfo(BaseStats):
    ep_stats: None | EnvStats = None
    cpu_time: None | float = None
    list_nodes: None | list[Any, ...] = None
    success: None | bool = None

    @classmethod
    def new(cls, ep_stats, cpu_time, list_nodes, success):
        stats = cls()
        stats.ep_stats = ep_stats
        stats.cpu_time = cpu_time
        stats.list_nodes = list_nodes
        stats.success = success
        return stats


