import time
from multiprocessing import cpu_count

import matplotlib.ticker as ticker
import mrl
import pandas as pd
import pybullet as p
import seaborn as sns
from matplotlib import pyplot as plt
from mrl.algorithms.continuous_off_policy import ActorPolicy, DDPG
from mrl.configs.continuous_off_policy import protoge_config
from mrl.modules.action_noise import ContinuousActionNoise
from mrl.modules.curiosity import DensityAchievedGoalCuriosity
from mrl.modules.density import RawKernelDensity
from mrl.modules.env import EnvModule
from mrl.modules.eval import EpisodicEval
from mrl.modules.goal_reward import GoalEnvReward
from mrl.modules.logging import Logger
from mrl.modules.model import PytorchModel
from mrl.modules.normalizer import Normalizer, MeanStdNormalizer
from mrl.modules.train import StandardTrain
from mrl.replays.online_her_buffer import OnlineHERBuffer
from mrl.utils.misc import make_activ
from mrl.utils.networks import Actor, FCBody, Critic
from mrl.utils.random_process import GaussianProcess
from mrl.utils.schedule import ConstantSchedule
from rich.progress import track
from torch import nn

from rrt_ml.algorithms.base import *
from rrt_ml.environments.car_navigation_bullet_env import *
from rrt_ml.utilities.analytic import *
from rrt_ml.utilities.configs import *
from rrt_ml.utilities.formulas import *
from rrt_ml.utilities.hints import *
from rrt_ml.utilities.infos import *
from rrt_ml.utilities.paths import *
from rrt_ml.utilities.stats import *


class RL(Algorithm):

    def __init__(self, cfg: MasterConfig):
        """
        Initialize.
        :param cfg: configuration
        """

        super(RL, self).__init__(cfg)
        self.agent = None  # type: None | mrl.Agent
        self.eval_env = None  # type: None | CarNavigationBulletEnv
        self.mrl_config = None

        # Setup
        self._setup()

    def train(self):
        """
        Train agent.
        """

        # No need if already trained
        if self.training_over:
            return

        # Get total num of epochs
        epochs_total = int(self.cfg.rl.train.n_timesteps // self.cfg.rl.val.interval)

        # Train loop
        for epoch_n in track(range(self.epoch + 1, epochs_total), "Training RL agent..."):
            # Train for a number of timesteps
            self.agent.train(num_steps=self.cfg.rl.val.interval)

            # Evaluate with MRL (random resets) and custom (deterministic reset)
            rand_val_reward = np.mean(self.agent.eval(num_episodes=self.cfg.rl.val.n_episodes).rewards)
            env_stats, det_val_reward = self._get_eval_stats()
            train_timestep_num = epoch_n * self.cfg.rl.val.interval

            # Get and save epoch info
            epoch_info = RLEpochInfo.new(det_val_reward, rand_val_reward, env_stats, train_timestep_num)
            self._set_attrs_after_epoch(epoch_info)

            # Save
            self.save_stats(epoch_info)
            self.save_checkpoint(epoch_info)

            # Log
            self.log_console(epoch_info)
            self.log_tensorboard(epoch_info)

        # Set training flag and save attrs
        self.training_over = True
        self._set_save_attributes()

    def test(self):
        """
        Test RL.
        """

        self._test_plot_entropy()
        self._test_plot_rewards()
        self._test_plot_progress()
        self._test_plot_horizon_success_rate()
        self._test_plot_multiple_problems()
        self._test_plot_compare_mpcrs()
        self._test_plot_compare_rs()
        self._test_plot_compare_rs_mpcrs()
        self._test_chasing()

    def save_checkpoint(self, epoch_info: 'RLEpochInfo'):
        """
        Save checkpoint. Equal to base method except for last line.
        """

        # Save attributes
        self._set_save_attributes()

        # Save agent if improved deterministic and random evaluations
        if self.model_curr_loss > self.model_best_loss:
            # Log
            self.console.print(f"\n[red bold underline]Total reward increased from "
                               f"{self.model_best_loss:.2f} to {self.model_curr_loss:.2f}... Saving model...[/]")

            # Save agent
            self.agent.save("checkpoint")

    def save_stats(self, epoch_info: 'RLEpochInfo'):
        """
        Set up stats to save.
        """

        self.stats.env_stats.append(epoch_info.env_stats)
        self.stats.det_val_rewards.append(epoch_info.det_val_reward)
        self.stats.rand_val_rewards.append(epoch_info.rand_val_reward)
        self.stats.total_val_rewards.append(epoch_info.det_val_reward + epoch_info.rand_val_reward)
        self.stats.train_timestep_nums.append(epoch_info.train_timestep_num)

        self.stats.save_to_file(self.path_stats)

    def log_console(self, epoch_info: 'RLEpochInfo'):
        """
        Log to console.
        :param epoch_info: current epoch info
        """

        tt = epoch_info.train_timestep_num
        rr = epoch_info.rand_val_reward
        dr = epoch_info.det_val_reward
        self.console.print(
            f"\n[blue bold underline]Train Timestep:[/blue bold underline] [blue]{tt}[/blue]\t"
            f"[cyan bold underline]Current Rewards (rand/det):[/cyan bold underline] [cyan]{rr}/{dr}[/cyan]\t"
        )

    def log_tensorboard(self, epoch_info: 'RLEpochInfo'):
        """
        For consistency only, handled by MRL.
        """
        pass

    def get_distance_data(self, num_episodes=100000):
        """
        Get distance data for training supervised models.
        """

        # Create files and folders
        train_file_path, val_file_path, test_file_path = self._get_distance_data_files_paths()

        # Set agent to eval mode
        self.agent.eval_mode()

        # Create env with modified config and reset master to get all stats
        cfg = self._get_distance_env_cfg()
        env = CarNavigationBulletEnv(cfg)

        # Loop episodes
        ep_counter = 0
        for i in range(num_episodes):

            # Get data
            data = []
            l_obs = []
            self._get_episode_info_recursive(env, l_obs, data)

            # Increment counter
            ep_counter += len(data)

            # Save to file
            if ep_counter % 2 == 0:

                # Save batch
                self._set_save_distance_batch(data, train_file_path, val_file_path, test_file_path)

                # Log
                self.console.print(
                    f"\n[blue bold underline]Generating distance data...[/blue bold underline]\t"
                    f"[cyan bold underline]Number of episodes:[/cyan bold underline] [cyan]{ep_counter}[/cyan]\t"
                )

    def get_action(self, obs: dict[str, Vector]) -> Vector2:
        """
        Get agent prediction.
        :return: action
        """

        self.agent.eval_mode()

        return self.agent.policy(obs).flatten()

    def _get_distance_data_files_paths(self):
        """
        Create files and folders to hold data.
        :returns: train and validation csv paths
        """

        # Set paths
        folder_path = Paths().data_rl_distance / self.cfg.general.config_name_or_prefix
        train_file_path = folder_path / 'train.csv'
        val_file_path = folder_path / 'val.csv'
        test_file_path = folder_path / 'test.csv'

        # Create files and folders
        if not folder_path.exists():
            folder_path.mkdir()
        if not train_file_path.exists():
            with open(str(train_file_path), "w") as _:
                pass
            with open(str(val_file_path), "w") as _:
                pass
            with open(str(test_file_path), "w") as _:
                pass

        return train_file_path, val_file_path, test_file_path

    def _get_distance_env_cfg(self):
        """
        Change env config to generate distance data.
        :return: new master config
        """

        # Change config
        cfg = MasterConfig.load_from_experiment('rl', self.cfg.rrt.names.rl)
        cfg.env.car.pose = 0
        cfg.env.target.pose = 5
        cfg.env.general.max_timestep = 100
        cfg.env.general.seed = None
        cfg.env.general.stats = True
        cfg.env.general.gui = False

        return cfg

    def _get_eval_stats(self) -> tuple[EnvStats, float]:
        """
        Eval agent on episodes with deterministic resetting and capture stats.
        :return: stats and accumulated reward
        """

        # Set agent to eval mode (don't use dropout, etc)
        self.agent.eval_mode()

        # Master reset to set the seed and reset deterministically
        self.eval_env.reset_master()

        # Episodes loops
        sum_rewards = 0
        for i in range(self.cfg.rl.val.n_episodes):
            done = False
            obs = self.eval_env.reset()
            while not done:
                obs, reward, done, _ = self.eval_env.step(self.agent.policy(obs).flatten())
                sum_rewards += reward

        # Get stats
        return self.eval_env.stats, sum_rewards / self.cfg.rl.val.n_episodes

    def _get_episode_info_recursive(self, env: CarNavigationBulletEnv, l_obs: list[dict, ...], data: list[np.ndarray, ...]):
        """
        Get episode info recursively.
        :param env:
        :param l_obs:
        :param data:
        :return:
        """

        # If there are no list of observations we need to get it
        if len(l_obs) == 0:

            # If there is data we need to return it
            if len(data) > 0:
                return data

            # Place target at random pose
            env.reset_master()
            env.reset()

            # Place car at origin
            env.car.set_pose([0, 0, 0])

            # Get target obs
            obs = env._get_observation()
            car_state = obs['achieved_goal']
            target_state = obs['desired_goal']

            # Play episode
            done, done_info = env._get_done_indicator(env.compute_reward(car_state, target_state))
            info = {'done_info': done_info}
            while not done:

                # Policy action
                action = self.agent.policy(obs).flatten()

                # Step
                obs, reward, done, info = env.step(action)
                
                # Add to list of obs
                l_obs.append(obs)

            # Add to database
            success = info['done_info']['success']
            data.append(np.concatenate((
                target_state,
                np.array([success]),
                np.array([env.stats.get_time_to_reach(0)]),
                np.array([env.timestep]),
                np.array([env.stats.get_distance_traveled(0)]))
            ))

            self._get_episode_info_recursive(env, l_obs, data)

        else:

            # Reset to get stats and zero joints
            env.reset_master()
            env.reset()

            # Transform to origin
            env.car.set_pose([0, 0, 0])
            env.target.set_pose(transform_to_origin(l_obs.pop(0)))
            
            # Get new target pose
            obs = env._get_observation()
            car_state = obs['achieved_goal']
            target_state = obs['desired_goal']

            # Play episode
            done, done_info = env._get_done_indicator(env.compute_reward(car_state, target_state))
            info = {'done_info': done_info}
            while not done:

                # Policy action
                action = self.agent.policy(obs).flatten()

                # Step
                obs, reward, done, info = env.step(action)

            # Add to database
            success = info['done_info']['success']
            data.append(np.concatenate((
                    target_state,
                    np.array([success]),
                    np.array([env.stats.get_time_to_reach(0)]),
                    np.array([env.timestep]),
                    np.array([env.stats.get_distance_traveled(0)]))
                ))

            # Replay until there are no intermediate observations
            self._get_episode_info_recursive(env, l_obs, data)
            
            return data

    def _set_attrs_after_epoch(self, epoch_info):
        """
        Set basic attributes after an epoch o training.
        :param epoch_info: current epoch info
        """

        # Update iteration
        self.epoch = int(100 * epoch_info.train_timestep_num / self.cfg.rl.train.n_timesteps)

        # Update current loss
        self.model_curr_loss = epoch_info.det_val_reward + epoch_info.rand_val_reward

        # Check best loss so far
        if len(self.stats.total_val_rewards) > 1:
            self.model_best_loss = max(self.stats.total_val_rewards)

    def _set_save_distance_batch(self, data: list, train_file_path: str, val_file_path: str, test_file_path: str):
        """
        Save batch of distance data
        :param data: list of rows with x and y pairs
        :param train_file_path: path to train_file
        :param val_file_path: path to val_file
        """

        # Ignore warnings
        self._set_ignore_static_warnings()

        # Generate 10% of data as validation and 20% as test
        rand = np.random.rand()
        if rand < 0.1:
            with open(str(val_file_path), "a") as file:
                # noinspection PyTypeChecker
                np.savetxt(file, np.array(data), delimiter=",")
        elif rand < 0.25:
            with open(str(test_file_path), "a") as file:
                # noinspection PyTypeChecker
                np.savetxt(file, np.array(data), delimiter=",")
        else:
            with open(str(train_file_path), "a") as file:
                # noinspection PyTypeChecker
                np.savetxt(file, np.array(data), delimiter=",")

    def _test_chasing(self):
        """
        Test chasing different poses.
        """

        # Env
        cfg = self.cfg.copy(deep=True)
        cfg.env.general.gui = True
        cfg.env.car.pose = [0, 0, 0]
        cfg.env.target.pose = 1
        cfg.env.reward.epsilon = -0.15
        cfg.env.general.max_timestep = 10000
        env = CarNavigationBulletEnv(cfg)
        obs = env.reset()

        # Loop all episodes
        for _ in range(100):

            # Loop
            time.sleep(0.5)
            done = False
            pose = np.random.uniform([-1, -1, 0], [1, 1, 2*np.pi])
            env.target.set_pose(pose)
            while not done:
                obs, reward, done, info = env.step(self.get_action(obs))
                time.sleep(1/45)

    def _test_plot_compare_mpcrs(self):
        """
        Compare RL and MPC with Reed-Shepp paths.
        """

        # Return if figure exists
        path = self.path_figs / 'rl_comparacao_rsmpc.png'
        if path.exists():
            return

        # Change config
        cfg = self.cfg.copy(deep=True)
        cfg.env.general.seed = 1
        cfg.env.general.gui = True
        cfg.env.general.stats = True
        cfg.env.target.pose = 1.5
        cfg.env.car.pose = [0, 0, 0]

        # Initialize env and reset master
        env = CarNavigationBulletEnv(cfg)
        env.reset_master()

        # Initialize figure
        fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row', figsize=(12, 9.6))
        fig.subplots_adjust(hspace=-0.2)

        # Episode loop for RL
        for ep in range(3):

            # Get episode info for RL
            done, obs = False, env.reset()
            while not done:
                obs, reward, done, info = env.step(self.get_action(obs))

            # Get car info to place ghosts and actions
            car_poses = env.stats.bullet.car_poses[ep, ...]
            car_steers = env.stats.bullet.car_steers[ep, ...]

            # Remove all zeros rows
            car_poses = car_poses[~np.all(car_poses == 0, axis=1)]
            car_steers = car_steers[~np.all(car_steers == 0, axis=1)]

            # Get indexes to place ghosts
            interval = cfg.env.test.general.ghost_interval
            n_timesteps = car_poses.shape[0]
            idxs = np.arange(0, n_timesteps, interval)

            # Place ghosts
            for pose, steer in zip(car_poses[idxs, :], car_steers[idxs, :]):
                env._set_place_ghost_car(pose, steer)

            # Place car at final pose
            env.car.set_pose(car_poses[-1, :])

            # Take screenshot and clear ghosts
            img = env._get_image_current_top_view()
            env._set_remove_ghosts()

            # Plot on axis
            axs[0, ep].imshow(img, origin='lower', extent=cfg.env.test.none.img_extent)

            # Get distance traveled and time
            length = env.stats.get_distance_traveled(ep)
            time = env.stats.get_time_to_reach(ep)
            axs[0, ep].set_title(f'Agente: {time:.2f}s / {length:.2f}m')

            # Set y axis
            if ep == 0:
                axs[0, ep].set_ylabel('y [m]')

        # Now get results for RSMPC, change config and re-init env
        cfg.env.car.f_val = 240
        cfg.env.general.max_timestep = 2000
        p.disconnect(physicsClientId=env.client)
        del env
        env = CarNavigationBulletEnv(cfg)
        env.reset_master()

        # Change config and initialize MPCRS
        rsmpc = MPCRS(cfg)

        # Episode loop for MPCRS
        for ep in range(3):

            # Get episode info for RL
            done, obs = False, env.reset()
            while not done:
                action = rsmpc.get_action(obs)
                obs, reward, done, info = env.step(action)

            # Get car info to place ghosts and actions
            car_poses = env.stats.bullet.car_poses[ep, ...]
            car_steers = env.stats.bullet.car_steers[ep, ...]

            # Remove all zeros rows
            car_poses = car_poses[~np.all(car_poses == 0, axis=1)]
            car_steers = car_steers[~np.all(car_steers == 0, axis=1)]

            # Get indexes to place ghosts
            interval = cfg.env.test.general.ghost_interval
            n_timesteps = car_poses.shape[0]
            idxs = np.arange(0, n_timesteps, interval)

            # Place ghosts
            for pose, steer in zip(car_poses[idxs, :], car_steers[idxs, :]):
                env._set_place_ghost_car(pose, steer)

            # Place car at final pose
            env.car.set_pose(car_poses[-1, :])

            # Take screenshot and clear ghosts
            img = env._get_image_current_top_view()
            env._set_remove_ghosts()

            # Plot on axis
            axs[1, ep].imshow(img, origin='lower', extent=cfg.env.test.none.img_extent)

            # Get distance traveled and time
            length = env.stats.get_distance_traveled(ep)
            time = env.stats.get_time_to_reach(ep)
            axs[1, ep].set_title(f'CPM: {time:.2f}s / {length:.2f}m')

            # Set x axis
            axs[1, ep].set_xlabel('x [m]')

            # Set y axis
            if ep == 0:
                axs[1, ep].set_ylabel('y [m]')

        # Delete env
        p.disconnect(physicsClientId=env.client)

        # Save figure
        fig.savefig(str(path), dpi=600, bbox_inches='tight')

    def _test_plot_compare_rs(self):
        """
        Compare RL path with RS path.
        """

        # Return if figure exists
        path = self.path_figs / 'rl_comparacao_rs.png'
        if path.exists():
            return

        # Change config
        cfg = self.cfg.copy(deep=True)
        cfg.env.general.seed = 1
        cfg.env.general.gui = True
        cfg.env.general.stats = True
        cfg.env.target.pose = 1.5
        cfg.env.car.pose = [0, 0, 0]
        cfg.rrt.rs.curvature = 2.5

        # Initialize env and reset master
        env = CarNavigationBulletEnv(cfg)
        env.reset_master()

        # Initialize figure
        fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row', figsize=(12, 9.6))
        fig.subplots_adjust(hspace=-0.2)

        # Episode loop for RL
        ep = 0
        for i, _ in enumerate(axs):
            for j, _ in enumerate(axs[i, :]):

                # Reset env
                done, obs = False, env.reset()

                # Get RS path length before changing env
                rs = MPCRS(cfg)
                rs._set_rs_path(obs)
                length_rs = rs.get_distance(Node(env.car.get_pose()), Node(env.target.get_pose()))

                while not done:
                    obs, reward, done, info = env.step(self.get_action(obs))

                # Get car info to place ghosts and actions
                car_poses = env.stats.bullet.car_poses[ep, ...]
                car_steers = env.stats.bullet.car_steers[ep, ...]

                # Remove all zeros rows
                car_poses = car_poses[~np.all(car_poses == 0, axis=1)]
                car_steers = car_steers[~np.all(car_steers == 0, axis=1)]

                # Get indexes to place ghosts
                interval = cfg.env.test.general.ghost_interval
                n_timesteps = car_poses.shape[0]
                idxs = np.arange(0, n_timesteps, interval)

                # Place ghosts
                for pose, steer in zip(car_poses[idxs, :], car_steers[idxs, :]):
                    env._set_place_ghost_car(pose, steer)

                # Place car at final pose
                env.car.set_pose(car_poses[-1, :])

                # Take screenshot and clear ghosts
                img = env._get_image_current_top_view()
                env._set_remove_ghosts()

                # Plot on axis
                axs[i, j].imshow(img, origin='lower', extent=cfg.env.test.none.img_extent)

                # Plot RS path
                axs[i, j].plot(rs.rs_xs, rs.rs_ys, color='yellow', lw=2)

                # Get distance traveled
                length_rl = env.stats.get_distance_traveled(ep)

                # Set title as lengths
                axs[i, j].set_title(f'Agente: {length_rl:.2f}m\nRS: {length_rs:.2f}m')

                # Set x axis
                if i == 1:
                    axs[i, j].set_xlabel('x [m]')

                # Set y axis
                if j in [0, 3]:
                    axs[i, j].set_ylabel('y [m]')

                # Increment episode counter
                ep += 1

        # Delete env
        p.disconnect(physicsClientId=env.client)

        # Save figure
        fig.savefig(path, dpi=600, bbox_inches='tight')

    def _test_plot_compare_rs_mpcrs(self):
        """
        Compare RL, RS and MPC path lengths.
        """

        # Return if figure exists
        path = self.path_figs / 'rl_comparacao_rs_rsmpc.png'
        if path.exists():
            return

        # Constants
        n_trials = 20

        # Change config
        cfg = self.cfg.copy(deep=True)
        cfg.env.general.seed = 1
        cfg.env.general.gui = True
        cfg.env.general.stats = True
        cfg.env.target.pose = 1.5

        # Initialize dataframe
        df = pd.DataFrame({
            'Critério': [],
            'Custo': [],
            'Semente': [],
            'Abordagem': []
        })

        # Prepare for MPC
        cfg.env.car.f_val = 240
        cfg.env.general.max_timestep = 2000

        # Episode loop for MPC
        env = CarNavigationBulletEnv(cfg)
        env.reset_master()
        mpcrs = MPCRS(cfg)
        success_ep_nums = []
        i = -1
        while len(success_ep_nums) < n_trials:

            # Increment
            i += 1

            # Get episode info for RL
            done, obs, info = False, env.reset(), {}
            while not done:
                action = mpcrs.get_action(obs)
                obs, reward, done, info = env.step(action)

            if info['done_info']['success']:
                success_ep_nums.append(i)

            # Add to dataframe
            df.loc[len(df)] = ['Distância', env.stats.get_distance_traveled(i), i, 'RS+CPM']
            df.loc[len(df)] = ['Tempo', env.stats.get_time_to_reach(i), i, 'RS+CPM']

        # Episode loop for RL
        p.disconnect(physicsClientId=env.client)
        del env
        cfg = self.cfg.copy(deep=True)
        cfg.env.general.seed = 1
        cfg.env.general.gui = True
        cfg.env.general.stats = True
        cfg.env.target.pose = 1.5
        env = CarNavigationBulletEnv(cfg)
        env.reset_master()
        i = -1
        for j in range(success_ep_nums[-1]+1):

            # Reset before maybe continuing
            done, obs = False, env.reset()

            # Increment
            i += 1

            # Continue if not success for MPC
            if i not in success_ep_nums:
                continue

            # Get episode info for RL
            while not done:
                obs, reward, done, info = env.step(self.get_action(obs))

            # Add to dataframe
            if i in success_ep_nums:
                df.loc[len(df)] = ['Distância', env.stats.get_distance_traveled(i), i, 'Agente']
                df.loc[len(df)] = ['Tempo', env.stats.get_time_to_reach(i), i, 'Agente']

        # Prepare for RS
        cfg.rrt.rs.curvature = 2.5
        cfg.rrt.rs.step_size = 1/240
        p.disconnect(physicsClientId=env.client)
        del env
        env = CarNavigationBulletEnv(cfg)

        # Change config and initialize RS
        env.reset_master()
        mpcrs = MPCRS(cfg)

        # Episode loop for RS
        i = -1
        for j in range(success_ep_nums[-1] + 1):

            # Reset before maybe continuing
            done, obs = False, env.reset()

            # Increment
            i += 1

            # Continue if not success for MPC
            if i not in success_ep_nums:
                continue

            # Get episode info for RL
            mpcrs._set_rs_path(obs)
            node_from = Node(env.car.get_pose())
            node_to = Node(env.target.get_pose())

            if i in success_ep_nums:
                df.loc[len(df)] = ['Distância', mpcrs.get_distance(node_from, node_to), i, 'RS']
                df.loc[len(df)] = ['Tempo', (1/240)*len(mpcrs.rs_xs), i, 'RS']

        # Delete env
        p.disconnect(physicsClientId=env.client)

        # Rename members
        df2 = df.replace('Distância', 'Distância (m)')
        df2 = df2.replace('Tempo', 'Tempo (s)')

        # Save figure
        sns.set(font_scale=1.2)
        sns.set_style('whitegrid')
        ax = sns.barplot(data=df2, x="Critério", y="Custo", hue='Abordagem')
        l = ax.legend()
        l.set_title('')
        ax.get_figure().savefig(str(path), dpi=600, bbox_inches='tight')

    def _test_plot_entropy(self):
        """
        Get entropy plot.
        """

        # Return if figure exists
        path = self.path_figs / 'rl_entropia.png'
        if path.exists():
            return

        # Get entropy csv path
        csv_path = next(self.path_figs.parent.glob('*entropy*'))

        # Read as dataframe
        df = pd.read_csv(str(csv_path))

        # Plot
        fig, ax = plt.subplots()
        ax.plot(df['step'][7:], df['Explore/ag_kde_entropy'][7:])
        ax.grid(True)
        ax.set_xlabel('Passos de tempo')
        ax.set_ylabel('Entropia')
        fig.savefig(str(path), dpi=600, bbox_inches='tight')

    def _test_plot_horizon_success_rate(self):
        """
        Plot success rate for different horizons.
        """

        # Return if figure exists
        path = self.path_figs / 'rl_horizonte_taxa_sucesso.png'
        if path.exists():
            return

        # Env without gui to go faster and car pose is always [0, 0, 0]
        cfg = self.cfg.copy(deep=True)
        cfg.env.general.gui = False
        cfg.env.car.pose = [0, 0, 0]

        # Initialize distances
        distances = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                     5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]

        # Loop through distances
        successes = []
        for d in distances:

            # Change env config distance and initialize
            cfg.env.target.pose = d
            env = CarNavigationBulletEnv(cfg)

            # Loop all episodes
            success = 0
            for _ in range(100):

                # Episode loop
                done, obs, info = False, env.reset(), {}
                while not done:
                    obs, reward, done, info = env.step(self.get_action(obs))

                # Check done reason
                if info['done_info']['success']:
                    success += 1

            # Delete env
            del env

            # Add to list
            successes.append(success)

            # Log
            self.console.print(f"\n[red bold underline]Done testing distance {d}...[/]")

        # Initialize figure and plot
        fig, ax = plt.subplots()  # type: plt.Figure, plt.Axes
        ax.bar([str(d) for d in distances], successes, color='blue', edgecolor='blue')

        # Rotate x-axis labels to fit better
        for label in ax.get_xticklabels():
            label.set_rotation(45)

        # Change y-axis to percent formatting
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())

        # Add grid, axis labels
        ax.grid(True)
        ax.set_xlabel('c [m]')
        ax.set_ylabel('Taxa de sucesso')

        # Save
        fig.savefig(str(path), dpi=600, bbox_inches='tight')

    def _test_plot_rewards(self):
        """
        Plot test reward curves.
        """

        # Return if figure exists
        path = self.path_figs / 'rl_recompensa.png'
        if path.exists():
            return

        # Get constants
        timesteps = self.stats.train_timestep_nums
        rewards = self.stats.total_val_rewards

        # Plot
        fig, ax = plt.subplots()
        ax.plot(timesteps, rewards)
        ax.grid(True)
        ax.set_xlabel('Passos de tempo')
        ax.set_ylabel('Recompensa')
        fig.savefig(path, dpi=600, bbox_inches='tight')

    def _test_plot_multiple_problems(self):
        """
        Plot agent actions in an episode.
        """

        # Return if figure exists
        path = self.path_figs / 'rl_diversos_problemas.png'
        if path.exists():
            return

        # Change config
        cfg = self.cfg.copy(deep=True)
        cfg.env.general.seed = 0
        cfg.env.general.gui = True
        cfg.env.general.stats = True
        cfg.env.target.pose = 1.5
        cfg.env.car.pose = [0, 0, 0]

        # Initialize env and reset master
        env = CarNavigationBulletEnv(cfg)
        env.reset_master()

        # Initialize figure
        fig, axs = plt.subplots(nrows=2, ncols=3, sharex='col', sharey='row', figsize=(12, 9.6))
        fig.subplots_adjust(hspace=-0.2)
        axs = axs.flatten()

        # Episode loop
        for ep in range(6):

            # Get episode info
            done, obs = False, env.reset()
            while not done:
                obs, reward, done, info = env.step(self.get_action(obs))

            # Get car info to place ghosts and actions
            car_poses = env.stats.bullet.car_poses[ep, ...]
            car_steers = env.stats.bullet.car_steers[ep, ...]

            # Remove all zeros rows
            car_poses = car_poses[~np.all(car_poses == 0, axis=1)]
            car_steers = car_steers[~np.all(car_steers == 0, axis=1)]

            # Get indexes to place ghosts
            interval = cfg.env.test.general.ghost_interval
            n_timesteps = car_poses.shape[0]
            idxs = np.arange(0, n_timesteps, interval)

            # Place ghosts
            for pose, steer in zip(car_poses[idxs, :], car_steers[idxs, :]):
                env._set_place_ghost_car(pose, steer)

            # Place car at final pose
            env.car.set_pose(car_poses[-1, :])

            # Take screenshot and clear ghosts
            img = env._get_image_current_top_view()
            env._set_remove_ghosts()

            # Plot on axis
            axs[ep].imshow(img, origin='lower', extent=cfg.env.test.none.img_extent)

            # Set titles
            axs[ep].set_title(f'Cenário {ep + 1}')

            # Set axis labels
            if ep in [0, 3]:
                axs[ep].set_ylabel('y [m]')
            if ep in [3, 4, 5]:
                axs[ep].set_xlabel('x [m]')

        # Disconnect env
        p.disconnect(physicsClientId=env.client)

        # Save figure
        fig.savefig(str(path), dpi=600, bbox_inches='tight')

    def _test_plot_progress(self):
        """
        Plot policy trajectories over training time.
        """

        # Return if figure exists
        path = self.path_figs / 'rl_progresso.png'
        if path.exists():
            return

        # Constants
        ep_idx = 0
        train_timesteps_idx = [1, 2, 3, 4, 5, 90]

        # Change config before creating env
        cfg = self.cfg.copy(deep=True)
        cfg.env.general.gui = True

        # Initialize env to get screenshot
        env = CarNavigationBulletEnv(cfg)

        # Initialize figure
        fig, axs = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(12, 9.6))
        fig.subplots_adjust(hspace=-0.2)
        axs = axs.flatten()

        # Loop to get subplots
        for i, idx in enumerate(train_timesteps_idx):

            # Get target pose
            desired_goal = self.stats.env_stats[idx].mdp.desired_goals[ep_idx, 0, ...]
            target_pose = state4_to_pose3(desired_goal)

            # Place target at desired pose
            env.target.set_pose(target_pose)

            # Get car info to place ghosts
            car_poses = self.stats.env_stats[idx].bullet.car_poses[ep_idx, ...]
            car_steers = self.stats.env_stats[idx].bullet.car_steers[ep_idx, ...]

            # Remove all zeros rows
            car_poses = car_poses[~np.all(car_poses == 0, axis=1)]
            car_steers = car_steers[~np.all(car_steers == 0, axis=1)]

            # Get indexes to place ghosts
            interval = cfg.env.test.general.ghost_interval
            n_timesteps = car_poses.shape[0]
            idxs = np.arange(0, n_timesteps, interval)

            # Place ghosts
            for pose, steer in zip(car_poses[idxs, :], car_steers[idxs, :]):
                env._set_place_ghost_car(pose, steer)

            # Place car at final pose
            env.car.set_pose(car_poses[-1, :])

            # Take screenshot
            img = env._get_image_current_top_view()

            # Plot on axis
            axs[i].imshow(img, origin='lower', extent=cfg.env.test.none.img_extent)

            # Set title
            axs[i].set_title(f'Época {idx}')

            # Set axis labels
            if i in [0, 3]:
                axs[i].set_ylabel('y [m]')
            if i in [3, 4, 5]:
                axs[i].set_xlabel('x [m]')

            # Clear ghosts
            env._set_remove_ghosts()

        # Disconnect env
        p.disconnect(physicsClientId=env.client)

        # Save figure
        fig.savefig(str(path), dpi=600, bbox_inches='tight')

    def _setup(self):
        """
        Setup.
        """

        # Set up agent must be called before everything
        self._setup_agent()

        # Set up base settings
        self._setup_paths()
        self._setup_folders()
        self._setup_save_config_to_file()
        self._setup_attrs_to_save()
        self._setup_init_base_attrs()
        self._setup_init_model()
        self._setup_checkpoint()
        self._setup_stats()
        self._setup_console()
        self._setup_tensorboard()

        # Set up RL settings
        self._setup_eval_env()
        self._setup_just_use_policy()

    def _setup_agent(self):
        """
        Setup mega default and merge with user configuration.
        """

        # Load default mega config
        config = protoge_config()

        # Differences from protoge
        config.action_l2_regularization = self.cfg.rl.actor.l2
        config.action_noise = self.cfg.rl.exploration.noise
        config.activ = self.cfg.rl.net.activ
        config.batch_size = self.cfg.rl.train.batch_size
        config.eexplore = self.cfg.rl.exploration.epsilon
        config.grad_value_clipping = self.cfg.rl.net.grad_value_clipping
        config.initial_explore = self.cfg.rl.exploration.initial
        config.layers = self.cfg.rl.net.layers
        config.replay_size = self.cfg.rl.train.replay_size
        config.target_network_update_freq = self.cfg.rl.target.update_freq
        config.warm_up = self.cfg.rl.exploration.warm_up
        config.her = self.cfg.rl.general.her
        config.optimize_every = self.cfg.rl.train.optimize_every
        config.replay_size = self.cfg.rl.train.replay_size
        config.target_network_update_freq = self.cfg.rl.target.update_freq

        # Experiments folder
        config.parent_folder = Paths().experiments_rl

        # Agent name
        config.agent_name = str(self.cfg.general.config_name_or_prefix)

        # Parallel
        if self.cfg.rl.train.n_envs is None:
            config.num_envs = max(cpu_count() - 2, 1)
        else:
            config.num_envs = self.cfg.rl.train.n_envs
        if self.cfg.rl.val.n_envs is None:
            config.num_eval_envs = 1
        else:
            config.num_eval_envs = self.cfg.rl.val.n_envs

        # No parallel if testing?
        if not self.cfg.general.is_train:
            config.num_envs = 1
            config.num_eval_envs = 1

        # Train
        config.train_timestep = self.cfg.rl.train.n_timesteps

        # Setup and add basic modules to the config
        config.update(
            dict(
                trainer=StandardTrain(),
                evaluation=EpisodicEval(),
                policy=ActorPolicy(),
                logger=Logger(),
                state_normalizer=Normalizer(MeanStdNormalizer()),
                replay=OnlineHERBuffer(),
            )
        )

        # Discount factor
        if config.gamma < 1.0:
            config.clip_target_range = (np.round(-(1 / (1 - config.gamma)), 2), 0.0)
        if config.gamma == 1:
            config.clip_target_range = (np.round(-self.cfg.env.general.max_timestep - 5, 2), 0.0)

        # Prioritized experience replay
        config.prioritized_mode = "none"

        # Curiosity - ag density estimation
        config.ag_kde_tophat = RawKernelDensity(
            "ag",
            optimize_every=100,
            samples=10000,
            kernel="tophat",
            bandwidth=0.2,
            tag="_tophat",
        )
        config.ag_kde = RawKernelDensity(
            "ag",
            optimize_every=1,
            samples=10000,
            kernel="gaussian",
            bandwidth=0.1,
            log_entropy=True,
        )
        config.ag_curiosity = DensityAchievedGoalCuriosity(
            max_steps=self.cfg.env.general.max_timestep,
            num_sampled_ags=100,
            use_qcutoff=True,
            keep_dg_percent=-0.1,
        )

        # Actor noise?
        config.action_noise = ContinuousActionNoise(
            GaussianProcess, std=ConstantSchedule(config.action_noise)
        )

        # Off-policy model
        config.algorithm = DDPG()

        # Change target pose and gui (for train and val is always off)
        cfg = self.cfg.deep_copy_change('env.target.pose', self.cfg.rl.val.target_pose)
        cfg = cfg.deep_copy_change('env.general.gui', False)

        # Set up train environment
        def train_env_fn():
            return CarNavigationBulletEnv(cfg)

        # Set up train environment
        def val_env_fn():
            return CarNavigationBulletEnv(cfg)

        config.train_env = EnvModule(
            train_env_fn, num_envs=config.num_envs
        )
        config.eval_env = EnvModule(
            val_env_fn,
            num_envs=config.num_eval_envs,
            name="eval_env",
        )

        # Setup env success & done
        config.first_visit_succ = True
        config.first_visit_done = False

        # Setup algorithm
        config.algorithm = DDPG()

        # Setup layer normalization
        layer_norm_or_not = nn.LayerNorm if self.cfg.rl.net.layer_norm else nn.Identity

        # Setup and add the networks to the config
        e = config.eval_env
        config.actor = PytorchModel(
            "actor",
            lambda: Actor(
                FCBody(
                    e.state_dim + e.goal_dim,
                    self.cfg.rl.net.layers,
                    layer_norm_or_not,
                    make_activ(config.activ),
                ),
                e.action_dim,
                e.max_action,
            ),
        )
        config.critic = PytorchModel(
            "critic",
            lambda: Critic(
                FCBody(
                    e.state_dim + e.goal_dim + e.action_dim,
                    self.cfg.rl.net.layers,
                    layer_norm_or_not,
                    make_activ(config.activ),
                ),
                1,
            ),
        )

        # Intrinsic reward
        config.goal_reward = GoalEnvReward()

        # Return agent and complete config
        self.agent = mrl.config_to_agent(config)
        self.mrl_config = config

    def _setup_best_rewards(self):
        """
        Set up best rewards.
        """

        self.best_det_val_reward = -np.inf
        self.best_rand_val_reward = -np.inf

    def _setup_checkpoint(self):
        """
        Set up checkpoint.
        """

        # Check if there is a checkpoint
        self.load_checkpoint = self._get_checkpoint_exists_indicator()

        if self.load_checkpoint:
            self._set_load_attributes()

    def _setup_eval_env(self):
        """
        Set up evaluation env with deterministic resetting.
        """

        # Change config before creating env
        cfg = self.cfg.deep_copy_change('env.general.gui', self.cfg.rl.val.gui)
        cfg = cfg.deep_copy_change('env.general.stats', self.cfg.rl.val.stats)

        self.eval_env = CarNavigationBulletEnv(cfg=cfg)

    def _setup_init_base_attrs(self):
        """
        Initialize base attributes.
        """

        self.epoch = 0
        self.model_best_loss = -np.inf
        self.model_curr_loss = -np.inf

    def _setup_just_use_policy(self):
        """
        If loading algorithm just to use the policy we delete all envs.
        """

        # Disconnect all envs used for training, validation and testing
        if self.cfg.rl.general.just_use_policy:

            # Disconnect from eval env
            p.disconnect(self.eval_env.client)

            # Disconnect from train env module
            for env in self.agent.env.env.envs:
                p.disconnect(env.env.client)

            # Disconnect from eval env module
            for env in self.agent.eval_env.env.envs:
                p.disconnect(env.env.client)

    def _setup_stats(self):
        """
        Set up stats.
        """

        if self.load_checkpoint:
            try:
                self.stats = RLStats.load_from_file(self.path_stats)
            except EOFError:
                self.stats = RLStats.new(self.cfg)
                self.stats.save_to_file(self.path_stats)
        else:
            self.stats = RLStats.new(self.cfg)
            self.stats.save_to_file(self.path_stats)

    def _setup_tensorboard(self):
        """
        For consistency only, handled by MRL.
        """

        pass
