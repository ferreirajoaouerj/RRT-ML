from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
from joblib import dump, load
from pydantic import BaseModel
from scipy.spatial.distance import euclidean

from rrt_ml.utilities.configs import *
from rrt_ml.utilities.hints import *
from rrt_ml.utilities.misc import *


class BaseStats(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    z: None = None

    def save_to_file(self, path: Path | str):
        """
        Save pickle to file.
        """

        # Convert to string if needed
        if isinstance(path, Path):
            path = str(path)

        # Get dict and save to file
        d = self.dict()
        with open(path, 'wb') as f:
            dump(d, f)

        # Split possibly large pickle file
        self._split(path)

        # Delete possibly large pickle file
        Path(path).unlink(missing_ok=True)

    @classmethod
    def load_from_file(cls, path: Path):
        """
        Load from file.
        """

        # Join split files
        cls()._join(str(path))

        # Instantiate class from file dict
        with open(str(path), 'rb') as f:
            d = load(f)
        new = cls(**d)

        # Remove stats file created to load file
        path.unlink(missing_ok=True)

        return new

    @staticmethod
    def _split(path: str):
        """
        Split large file into smaller parts.
        """

        # Open the source file in binary mode
        input_file = open(path, 'rb')

        # Start file counter
        counter = 0

        # Loop read chunks and write to file
        while True:

            # Read a portion of the input file
            chunk = input_file.read(96000000)

            # End the loop if we have hit EOF
            if not chunk:
                break

            # Increment counter
            counter += 1

            # Create a new file name
            filename = path + str(counter).zfill(2)

            # Create a destination file
            dest_file = open(filename, 'wb')

            # Write to this portion of the destination file and close the file
            dest_file.write(chunk)
            dest_file.close()

        # Close the file being read
        input_file.close()

    @staticmethod
    def _join(path: str):
        """
        Join split pickled files.
        """

        # Create a new destination file
        output_file = open(path, 'wb')

        # Get a list of the file parts
        parts = glob(path + '[0-9][0-9]')

        # Go through each portion one by one
        for file in parts:

            # Assemble the full path to the file
            path = file

            # Open the part
            input_file = open(path, 'rb')

            # Read and merge
            while True:

                # Read all bytes of the part
                b = input_file.read(96000000)

                # Break out of loop if we are at end of file
                if not b:
                    break

                # Write the bytes to the output file
                output_file.write(b)

            # Close the input file
            input_file.close()

        # Close the output file
        output_file.close()


class EnvStats(BaseStats):

    class _bullet(BaseStats):
        car_poses: np.ndarray | None = None  # ep_n x timestep x pose (3)
        car_steers: np.ndarray | None = None  # ep_n x timestep x position (2)
        car_velocities: np.ndarray | None = None  # ep_n x timestep x velocity (1)
        dones: np.ndarray | None = None  # ep_n x timestep x done (bool, dict[str, bool])
        rewards: np.ndarray | None = None  # ep_n x timestep x reward (1)
        target_poses: np.ndarray | None = None  # ep_n x timestep x pose (3)
        time: np.ndarray | None = None  # ep_n x timestep x time (1)
        wall_time: np.ndarray | None = None  # ep_n x timestep x wall_time (1)

    class _mdp(BaseStats):
        achieved_goals: np.ndarray | None = None  # ep_n x timestep x goal (4)
        actions: np.ndarray | None = None  # ep_n x timestep x action (2)
        desired_goals: np.ndarray | None = None  # ep_n x timestep x goal (4)
        dones: np.ndarray | None = None  # ep_n x timestep x done (bool, dict[str, bool])
        next_achieved_goals: np.ndarray | None = None  # ep_n x timestep x goal (4)
        next_desired_goals: np.ndarray | None = None  # ep_n x timestep x goal (4)
        next_states: np.ndarray | None = None  # ep_n x timestep x state(6)
        rewards: np.ndarray | None = None  # ep_n x timestep x reward (1)
        states: np.ndarray | None = None  # ep_n x timestep x state (6)

    bullet: _bullet = _bullet()
    mdp: _mdp = _mdp()

    @classmethod
    def new(cls, cfg: MasterConfig):
        # Constants from config
        self = cls()
        n_episodes = cfg.rl.val.n_episodes
        n_mdp_timesteps = cfg.env.general.max_timestep
        n_bullet_timesteps = n_mdp_timesteps * (int(1 / (cfg.env.car.f_val * (1 / 240))))

        # MDP attributes
        self.mdp.states = np.zeros((n_episodes, n_mdp_timesteps, 6))
        self.mdp.achieved_goals = np.zeros((n_episodes, n_mdp_timesteps, 4))
        self.mdp.desired_goals = np.zeros((n_episodes, n_mdp_timesteps, 4))
        self.mdp.next_states = np.zeros((n_episodes, n_mdp_timesteps, 6))
        self.mdp.next_achieved_goals = np.zeros((n_episodes, n_mdp_timesteps, 4))
        self.mdp.next_desired_goals = np.zeros((n_episodes, n_mdp_timesteps, 4))
        self.mdp.actions = np.zeros((n_episodes, n_mdp_timesteps, 2))
        self.mdp.rewards = np.zeros((n_episodes, n_mdp_timesteps, 1))
        self.mdp.dones = np.empty((n_episodes, n_mdp_timesteps, 2), dtype=object)

        # Bullet attributes
        self.bullet.car_poses = np.zeros((n_episodes, n_bullet_timesteps, 3))
        self.bullet.car_steers = np.zeros((n_episodes, n_bullet_timesteps, 2))
        self.bullet.car_velocities = np.zeros((n_episodes, n_bullet_timesteps, 1))
        self.bullet.dones = np.empty((n_episodes, n_bullet_timesteps, 2), dtype=object)
        self.bullet.rewards = np.zeros((n_episodes, n_bullet_timesteps, 1))
        self.bullet.target_poses = np.zeros((n_episodes, n_bullet_timesteps, 3))
        self.bullet.time = np.zeros((n_episodes, n_bullet_timesteps, 1))
        self.bullet.wall_time = np.zeros((n_episodes, n_bullet_timesteps, 1))

        return self

    def get_distance_traveled(self, ep_num: int) -> float:
        """
        Calculate total distance traveled in an episode.
        :param ep_num: episode number
        :return: distance in meters
        """

        # Get xs and ys
        poses = self.bullet.car_poses[ep_num, ...]
        xs = poses[:, 0].flatten()
        ys = poses[:, 1].flatten()

        # Remove zeros
        xs = remove_trailing_zeros(xs)
        ys = remove_trailing_zeros(ys)

        # Sum differentials
        d = 0
        for i, (x, y) in enumerate(zip(xs[:-1], ys[:-1])):
            d += euclidean([x, y], [xs[i + 1], ys[i + 1]])

        return d

    def get_distance_traveled(self, ep_num: int) -> float:
        """
        Calculate total distance traveled in an episode.
        :param ep_num: episode number
        :return: distance in meters
        """

        # Get xs and ys
        poses = self.bullet.car_poses[ep_num, ...]
        xs = poses[:, 0].flatten()
        ys = poses[:, 1].flatten()

        # Remove zeros
        xs = remove_trailing_zeros(xs)
        ys = remove_trailing_zeros(ys)

        # Sum differentials
        d = 0
        for i, (x, y) in enumerate(zip(xs[:-1], ys[:-1])):
            d += euclidean([x, y], [xs[i + 1], ys[i + 1]])

        return d

    def get_time_to_reach(self, ep_num: int):
        """
        Get total time to reach goal in an episode.
        :param ep_num: episode number
        :return: total simulation time to reach
        """

        # Get list of times
        l_times = self.bullet.time[ep_num, :, 0]

        # Get last non-zero element of array (remove zeros and flat)
        return l_times.ravel()[np.flatnonzero(l_times)][-1]


class RLStats(BaseStats):
    det_val_rewards: list[float, ...] | None = None
    rand_val_rewards: list[float, ...] | None = None
    env_stats: list[EnvStats, ...] | None = None
    total_val_rewards: list[float, ...] | None = None
    train_timestep_nums: list[int, ...] | None = None

    @classmethod
    def new(cls, cfg):
        stats = cls()
        stats.det_val_rewards = []
        stats.rand_val_rewards = []
        stats.env_stats = []
        stats.total_val_rewards = []
        stats.train_timestep_nums = []
        return stats


class SLStats(BaseStats):
    arr_epoch_idx_state_dim: None | np.ndarray = None  # epoch x idx x sample_n x state
    train_loss: Vector | None = None
    train_kl_loss: Vector | None = None
    train_recon_loss: Vector | None = None
    val_loss: Vector | None = None
    val_kl_loss: Vector | None = None
    val_recon_loss: Vector | None = None

    @classmethod
    def new(cls, cfg: MasterConfig):
        stats = cls()
        n_epochs = cfg.sl.train.n_epochs
        num_val_maps = cfg.sl.val.n_maps
        num_gen_states = cfg.sl.val.n_states
        state_dim = cfg.sl.dim.state
        stats.arr_epoch_idx_state_dim = np.zeros(shape=(n_epochs, num_val_maps, num_gen_states, state_dim))
        stats.train_loss = []
        stats.train_kl_loss = []
        stats.train_recon_loss = []
        stats.val_loss = []
        stats.val_kl_loss = []
        stats.val_recon_loss = []
        return stats


class RRTStats(BaseStats):
    l_nodes: list[list[Any, ...], ...] | None = None
    l_wall_times: list[float, ...] | None = None

    @classmethod
    def new(cls, cfg):
        stats = cls()
        stats.l_nodes = []
        stats.l_wall_times = []
        return stats
