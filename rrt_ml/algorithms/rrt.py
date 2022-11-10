import copy
import time
from pathlib import Path
from typing import Callable, Optional

import matplotlib as mpl
import optuna
import pandas as pd
import seaborn as sns
from coral_pytorch.dataset import corn_label_from_logits
from matplotlib import pyplot as plt
from numpy.random import Generator
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
from tbparse import SummaryReader
from torch.utils.data import DataLoader
from torchensemble import SnapshotEnsembleRegressor
from torchensemble.utils import io
from torchensemble.utils.logging import set_logger

from rrt_ml.algorithms.base import *
from rrt_ml.algorithms.rl import *
from rrt_ml.algorithms.sl import *
from rrt_ml.environments.car_navigation_bullet_env import *
from rrt_ml.utilities.analytic import *
from rrt_ml.utilities.datasets import *
from rrt_ml.utilities.formulas import *
from rrt_ml.utilities.infos import *
from rrt_ml.utilities.maps import *
from rrt_ml.utilities.misc import *
from rrt_ml.utilities.models import *
from rrt_ml.utilities.paths import *
from rrt_ml.utilities.stats import *


class RRT(Algorithm):
    """
    RRT* algorithm with variations.
    """

    def __init__(self, cfg: MasterConfig):
        """
        Initialize.
        """

        super(RRT, self).__init__(cfg)
        self.controller_fn = None  # type: Optional[Callable[[dict], Vector2]]
        self.cutoff = None  # type: None | float | int
        self.dbg_map = None  # type: None | Map
        self.dbg_fig = None  # type: None | plt.Figure
        self.dbg_ax = None  # type: None | plt.Axes
        self.distance_fn = None  # type: Optional[Callable[[Node | list[Node, ...], Node], float]]
        self.env = None  # type: None | CarNavigationBulletEnv
        self.found_solution = None  # type: None | bool
        self.list_nodes = None  # type: None | list[Node, ...]
        self.list_sl_nodes = None  # type: None | list[Node, ...]
        self.min_distance = None  # type: None | float | int
        self.mpcrs = None  # type: None | MPCRS
        self.node_final = None  # type: None | Node
        self.node_init = None  # type: None | Node
        self.ordinal = None  # type: None | SnapshotEnsembleRegressor
        self.ordinal_range = None  # type: None | list[int, int]
        self.ordinal_scaler = None  # type: None | StandardScaler
        self.rng = None  # type: None | Generator
        self.rl = None  # type: None | RL
        self.rl_action_repeat = None  # type: None | int
        self.solution_simulation_info = None  # type: None | RRTSolutionInfo

        self._setup()

    def train(self):
        """
        Build RRT tree.
        """

        while len(self.list_nodes) < self.cfg.rrt.general.n_iter:

            # Sample final goal or leaner goal
            node_rand = self._get_random_node()

            # Check if we should check policy limitation
            if self.cfg.rrt.general.policy_limitation:
                if not self._get_within_policy_reach_indicator(node_rand):
                    continue

            # Get nearest node
            node_nearest = self._get_nearest_node(node_rand)

            # Get new node after simulating policy for a number of timesteps (this is the steer function)
            node_new = self._get_reach_info(node_nearest, node_rand, self.cfg.rrt.general.delta_t)

            # The new node may be 'none' in case of error, so we search another random node
            if node_new is None:
                continue

            # Debug callback
            self._debug_callback('new', locals())

            # Continue only if there is no collision until node new is reached
            if not node_new.info.done_info['collision']:

                # Get nodes near 'node_new'
                list_nodes_near = self._get_near_nodes(node_new)
                
                # Get the node with minimum cost from root
                node_min, node_with_info = self._get_min_cost_node(list_nodes_near, node_new)

                # Any error on finding 'node_min' we continue to the next iteration
                if node_min is None or node_with_info is None:
                    if not self.cfg.rrt.general.rewire:
                        node_min = node_nearest
                        node_with_info = node_new
                    else:
                        continue

                # Add node to graph
                self._set_add_node(node_min, node_new, node_with_info)

                # Debug callback
                self._debug_callback('add', locals())

                # Rewire the tree
                if self.cfg.rrt.general.rewire:
                    self._set_rewire(list_nodes_near, node_new)

            # Debug callback
            self._debug_callback('update', locals())
            
            # Save
            self.save_stats(None)
            self.save_checkpoint()

            # Log
            self.log_console()
            self.log_tensorboard()

        # Set training flag and save attrs
        self.training_over = True
        self._set_save_attributes()

    def test(self):
        """
        Test RRT.
        """

        # Get path to simulation info file
        path = Paths().exp_checkpoint('rrt', self.cfg.general.config_name_or_prefix) / 'simulation_info'

        # Return if simulation is already done
        path_part = path.parent / 'simulation_info01'
        if path_part.exists():
            self.solution_simulation_info = RRTSolutionInfo.load_from_file(path)
            return

        # Get list of waypoints
        list_nodes = self._get_solution()

        # Reset master to track stats on single episode (the first episode)
        self.env.reset_master()
        self.env.reset()

        # Place car at initial position
        self.env.car.set_pose(list_nodes[0].pose)

        # Start tracking cpu time
        cpu_time_start = time.time()

        # Navigate to next poses
        done, reward, info = False, -1, {}
        success = True
        for node in list_nodes[1:]:

            # Set target pose at next node
            self.env.target.set_pose(node.pose)

            # Control loop
            done, reward, info = False, -1, {}
            obs = self.env._get_observation()
            while not done:

                # Get action
                action = self.controller_fn(obs)

                # For RL, we use a lower frequency controller
                if self.cfg.rrt.general.controller == 'rl':

                    # Step 'n' times and update t
                    for _ in range(self.rl_action_repeat):
                        obs, reward, done, info = self.env.step(action)

                        # Break as soon as done
                        if done:
                            break

                # For MPC we step every bullet timestep
                else:
                    obs, reward, done, info = self.env.step(action)

            if info['done_info']['collision'] or info['done_info']['time']:
                success = False

        # Stop tracking cpu time
        cpu_time = time.time() - cpu_time_start

        # Set attribute and save to file
        self.solution_simulation_info = RRTSolutionInfo.new(self.env.stats.copy(deep=True), cpu_time, list_nodes,
                                                            success)
        self.solution_simulation_info.save_to_file(str(path))

        # Save new attributes
        self._set_save_attributes()

        # self._test_set_solution_simulation_info()
        # self._test_plot_solution_tree()
        # self._test_plot_solution_top_view()
        # self._test_get_solution_time()
        # self._test_get_solution_length()
        # self._test_plot_ordinal_accuracy_unique_min()
        # self._test_plot_ordinal_accuracy_top_x_unique_min()
        # self._test_plot_ordinal_accuracy_within_distance()

    def save_checkpoint(self):
        """
        Save attributes and models.
        """

        # # Saving the tree cost too much time, do it only every 50 iterations
        if len(self.list_nodes) % 50 == 0:
            self._set_save_attributes()

    def save_stats(self, epoch_info: RRTEpochInfo | None):
        """
        Save stats.
        """

        # Saving the tree cost too much time, do it only every 50 iterations
        if len(self.list_nodes) % 50 == 0:

            # Get tree
            l_nodes = copy.deepcopy(self.list_nodes)
            self.stats.l_nodes.append(l_nodes)

            # Get wall time
            self.stats.l_wall_times.append(time.time())

            # Save
            self.stats.save_to_file(self.path_stats)

    def log_console(self):
        """
        Log to console
        """

        # Log losses
        self.console.print(
            f"\n[blue bold underline]Epoch:[/] [blue]{len(self.list_nodes)}[/]\t"
        )

    def log_tensorboard(self):
        """
        Log to tensorboard
        """

        # Step is the number of nodes
        step = len(self.list_nodes)

        # Create map, add list of nodes and add figure
        mmap = Map(self.cfg)
        mmap.set_add_nodes(self.list_nodes)
        mmap.set_add_nodes(self.node_final)
        fig, _ = mmap.get_plot_tree()
        self.tb.add_figure("tree", fig, step)
        plt.close(fig)

        # Log solution found indicator (time to solve)
        self.tb.add_scalar("found_solution", int(self.found_solution), step)

        # If solution is found we track it's cost
        if self.found_solution:
            self.tb.add_scalar("solution_cost_length", self._get_solution_cost_length(), step)
            self.tb.add_scalar("solution_cost_time", self._get_solution_cost_time(), step)

    def _get_action_mpc(self, obs: dict):
        """
        Get action from pure pursuit controller.
        :param obs: current env observation (contains state and desired goal).
        :return: controller action.
        """

        return self.mpcrs.get_action(obs)

    def _get_action_rl(self, obs: dict):
        """
        Get action from agent's policy.
        :param obs: current env observation (contains state and desired goal).
        :return: controller action.
        """

        # Transform target
        new_target_pose = transform_to_origin(obs)
        obs['desired_goal'] = pose3_to_state4(new_target_pose)

        # Transform car 
        obs['achieved_goal'] = np.array([0, 0, 0, 1])
        obs['observation'][:4] = [0, 0, 0, 1]

        return self.rl.get_action(obs)

    def _get_distance_ordinal(self, node_from: Node | list[Node, ...], node_to: Node) -> Vector | float:
        """
        Get distance according to distance prediction ordinal model and maybe simulate some of them.
        :param node_from: initial node(s).
        :param node_to: final node.
        :return: distance value.
        """

        # Handle single instance case (default is parallel)
        if isinstance(node_from, Node):
            node_from = [node_from]

        # Hold input to the model (state and goal)
        n_rows = len(node_from)
        arr_goals = np.zeros((n_rows, 4))

        # Assign [xi yi si ci v phi xf yf sf cf]
        for i in range(n_rows):
            pose = transform_to_origin(None, node_from[i], node_to)
            arr_goals[i, :] = pose3_to_state4(pose)

        # Predict
        arr_goals = self.ordinal_scaler.transform(arr_goals)
        distance_pred = corn_label_from_logits(self.ordinal.predict(arr_goals)).detach().cpu().numpy().flatten()

        # Multiply by policy action repeat
        distance_pred *= self.rl_action_repeat
        
        # Check if we should simulate top-n %
        top_n_percent = int(self.cfg.rrt.general.distance_fn[-2:]) / 100
        top_n = round(top_n_percent * len(node_from))
        if top_n > 0 or self.cfg.rrt.general.distance_fn_mode == 'difference':

            # Sort ascending and get real distance for first 'n'
            distances_ascending_idxs = np.argsort(distance_pred)

            # Check the mode to select top-n nodes
            if self.cfg.rrt.general.distance_fn_mode == 'difference':
                min_dist = distance_pred[distances_ascending_idxs[0]]
                stop_at = 1
                for i, idx in enumerate(distances_ascending_idxs[1:]):
                    if np.abs(min_dist - distance_pred[idx]) > 10 * self.rl_action_repeat:
                        stop_at = i
                        break
            elif self.cfg.rrt.general.distance_fn_mode == 'difference':
                stop_at = top_n
            else:
                raise NotImplementedError

            # Sort ascending and get real distance for first 'n'
            for idx in distances_ascending_idxs[:stop_at]:

                # Check real distance
                node_new = self._get_reach_info(node_from[idx], node_to, 2000)

                # Update prediction with true value
                distance_pred[idx] = node_new.info.timesteps

        # return single number for single node distance calculation
        if distance_pred.size == 1:
            return distance_pred.item()
        else:
            return distance_pred.flatten()

    def _get_distance_rs(self, node_from: Node | list[Node, ...], node_to: Node) -> Vector | float:
        """
        Get distance according to Reed-Shepps path's length.
        :param node_from: initial node.
        :param node_to: final node.
        :return: distance value.
        """

        # Handle single distance case
        if isinstance(node_from, Node):
            node_from = [node_from]

        # Calculate distances
        distances = []
        for node in node_from:
            distances.append(self.mpcrs.get_distance(node, node_to))

        # return single number for single node distance calculation
        if len(distances) == 1:
            return distances[0]
        else:
            return np.array(distances)

    def _get_distance_sim(self, node_from, node_to):
        """
        Get distance according to simulation of agent's policy.
        :param node_from: initial node
        :param node_to: final node
        :return: distance value
        """

        # Handle single distance case
        if isinstance(node_from, Node):
            node_from = [node_from]

        # Get distance of all nodes
        distances = []
        for node in node_from:
            node_new = self._get_reach_info(node, node_to, 2000)
            distances.append(node_new.info.timesteps)

        # return single number for single node distance calculation
        if len(distances) == 1:
            return distances[0]
        else:
            return np.array(distances)

    def _get_min_cost_node(self, list_nodes_near, node_new) -> tuple[Node, Node]:
        """
        Get the node (from the list of near nodes) with minimum cost from root
        :param list_nodes_near: list nodes near 'node_new'
        :param node_new: the new node obtained after the simulation
        :return: the node with minimum cost
        """

        # Loop all nodes and track costs from root
        cost_min = np.inf
        node_min, node_final, node_with_info = None, None, None
        costs = []
        for node_near in list_nodes_near:
            
            # We simulate without time limit (we choose double the number of timesteps for near nodes)
            node_final = self._get_reach_info(node_near, node_new, 2000)

            # If the node could be reached and the cost is smaller
            if not node_final.info.done_info['collision'] and node_final.info.done_info['success']:
                cost = node_near.cost_from_root + node_final.info.cost
                costs.append(cost)
                if cost < cost_min:
                    cost_min = cost
                    node_min = node_near
                    node_with_info = node_final

            # Debug callback
            self._debug_callback('min', locals())

        return node_min, node_with_info

    def _get_near_nodes(self, node_new: Node) -> list[Node, ...]:
        """
        Get near nodes according to distance model and config cutoff value.
        :param node_new: near nodes in respect to this node
        :return: ordered list of near nodes (the nearest first)
        """

        # Get distance of all nodes to 'node_new'
        distances = self.distance_fn(self.list_nodes, node_new)

        # If there is only one distance we need wrap in a list for the next lines
        distances = np.array([distances]) if np.isscalar(distances) else distances

        # Get list of near nodes and near distances
        near_idxs = np.argwhere(np.array(distances) < self.cutoff).flatten()
        list_nodes_near = np.array(self.list_nodes)[near_idxs]
        distances_near = np.array(distances)[near_idxs]

        # Now we sort the distances, get the indexes and reorder the list of near nodes
        idxs_sorted_near_distances = np.argsort(distances_near)
        list_nodes_near_sorted = np.array(list_nodes_near)[idxs_sorted_near_distances]

        # Debug callback
        self._debug_callback('near', locals())

        return list_nodes_near_sorted

    def _get_nearest_node(self, node_rand):
        """
        Get the node that is nearest to the random node
        :param node_rand: random sampled node
        :return: nearest node
        """

        # Get distance of all nodes to 'node_new'
        distances = self.distance_fn(self.list_nodes, node_rand)
        
        # If there is only one distance we need wrap in a list for the next lines
        distances = np.array([distances]) if np.isscalar(distances) else distances
        
        # Get the index of the node with minimum distance
        nearest_idx = np.argmin(distances)

        # Debug callback
        self._debug_callback('nearest', locals())

        return self.list_nodes[nearest_idx]

    def _get_ordinal_config_and_path(self):
        """
        Get config with the best parameters for ordinal model.
        :return: the best config for ordinal model
        """

        # Get experiment checkpoint folder and config
        other_exp_name = self.cfg.rrt.ordinal.use_model_from_experiment
        if other_exp_name is not None:
            path_ckpt = Paths().exp_checkpoint('rrt', other_exp_name)
            cfg = MasterConfig.load_from_experiment('rrt', other_exp_name).copy(deep=True)
        else:
            path_ckpt = Paths().exp_checkpoint('rrt', self.cfg.general.config_name_or_prefix)
            cfg = self.cfg.copy(deep=True)

        # Check if optuna db exists
        db_path = path_ckpt / 'param_search' / 'study.db'
        db_exists = db_path.exists()

        # If it exists we need to load config and path to model from there
        if db_exists:

            # Load study
            path = 'sqlite:///' + str(db_path)
            study = optuna.create_study(study_name='study', storage=path, load_if_exists=True)

            # Get best params and set to config
            params = study.best_params
            cfg.rrt.ordinal.batch_size = params['batch_size']
            cfg.rrt.ordinal.layers = [params[f'n_units_{i}'] for i in range(params['n_layers'])]
            cfg.rrt.ordinal.n_estimators = params['n_estimators']

            # Get path to best model
            trial_num = study.best_trial.number
            model_path = str(db_path.parent / f'model_{str(trial_num).zfill(2)}')

        # Study does not exist, so we load from checkpoint folder or train from the beginning
        else:

            # If model exists at checkpoint folder we return this path
            if self._get_ordinal_model_exists_indicator(str(path_ckpt)):
                model_path = str(path_ckpt)

            # Otherwise, if search has to be performed the path depends on the best found
            else:

                # Perform search and assign new model path
                if self.cfg.rrt.ordinal.opt:
                    cfg, trial_num = self._get_ordinal_model_best_config_and_trial_num()
                    model_path = str(db_path.parent / f'model_{str(trial_num).zfill(2)}')

                # Don't search path is checkpoint folder
                else:
                    model_path = str(path_ckpt)

        return cfg, model_path

    def _get_ordinal_model_best_config_and_trial_num(self):
        """
        Search hyperparameters for the ordinal model and return the best config and trial number.
        :return: master config for best parameters
        """

        # Create path
        path = Paths().exp('rrt', self.cfg.general.config_name_or_prefix) / 'checkpoint' / 'param_search' / 'study.db'
        if not path.parent.exists():
            path.parent.mkdir()

        # Initialize study
        db_path = 'sqlite:///' + str(path)
        study = optuna.create_study(study_name='study', storage=db_path, load_if_exists=True)

        # Optimize if needed
        n_trials = 20
        path = path.parent / f'model_{n_trials - 1}'
        if not path.exists():
            ds_train = DistanceOrdinalDataset(self.cfg, 'train', None, None)
            ds_val = DistanceOrdinalDataset(self.cfg, 'val', ds_train.scaler, ds_train.timesteps_range)
            def objective(trial): return self._objective(trial, dict(cfg=self.cfg, ds_train=ds_train, ds_val=ds_val))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Copy and override attributes
        best_cfg = self.cfg.copy(deep=True)
        best_cfg.rrt.ordinal.batch_size = study.best_params['batch_size']
        best_cfg.rrt.ordinal.layers = [study.best_params[f'n_units_{i}'] for i in range(study.best_params['n_layers'])]
        best_cfg.rrt.ordinal.n_estimators = study.best_params['n_estimators']

        return best_cfg, study.best_trial.number

    def _get_ordinal_model_exists_indicator(self, model_path):
        """
        Check if there is a save for the best ordinal model
        :param model_path: path to model folder
        :return: boolean indicating whether the model exists
        """

        # Ignore warnings
        self._set_ignore_static_warnings()

        # Check for model existence
        if len(list(iter(Path(model_path).glob("*Ensemble*")))) > 0:
            return True
        else:
            return False

    def _get_parents_recursive(self, list_nodes: list[Node]) -> list[Node]:
        """
        Find the parents recursively.
        :return: find path going backwards.
        """

        # Get the last node
        last_pose = list_nodes[-1]

        # If last node has a parent we append it to the list and call the function again
        if last_pose.parent is not None:
            list_nodes.append(last_pose.parent)
            return self._get_parents_recursive(list_nodes)

        # If last node has no parent, it's the root node
        else:
            return list_nodes

    def _get_random_node(self) -> Node:
        """
        Get a random node: may be goal node or any random node.
        :return: chosen node.
        """

        # Get SL probability (constant or gaussian)
        if np.isscalar(self.cfg.rrt.sample.learner_prob):
            sl_prob = self.cfg.rrt.sample.learner_prob
        else:
            a, b, c = self.cfg.rrt.sample.learner_prob
            sl_prob = a*np.exp(-(len(self.list_nodes) - b)**2/(2*c**2))

        # Get goal probability
        goal_prob = self.cfg.rrt.sample.goal_prob if not self.found_solution else 0

        # Sampling weights and functions
        fns = [self._get_sample_final, self._get_sample_sl, self._get_sample_uniform]
        probs = [goal_prob, sl_prob, 1 - (goal_prob + sl_prob)]
        sample_fn = self.rng.choice(fns, p=probs)

        # Get random node
        random_node = sample_fn()

        return random_node

    def _get_reach_info(self, node_from, node_to, delta_t) -> None | Node:
        """
        Simulate trying to reach one node from others.
        :param node_from: origin node
        :param node_to: destination node
        :param delta_t: maximum simulation timesteps
        :return: timesteps to reach node, success indicator (no collision) and new node (very close to 'node_to').
        """

        # For MPC we can avoid wasting time with simulation (not 100% accurate) when checking distances and collisions
        if self.cfg.rrt.general.controller == 'mpc':

            # In this case we simulate the actions from the MPC controller trying to follow the RS path
            if self.cfg.rrt.mpc.simulate:
                node_new = self._get_reach_info_bullet(node_from, node_to, delta_t)

            # Here we assume the MPC controller follows the RS path with 100% fidelity
            else:
                node_new = self._get_reach_info_rs(node_from, node_to, delta_t)

        # For RL, we must simulate
        else:
            node_new = self._get_reach_info_bullet(node_from, node_to, delta_t)

        return node_new

    def _get_reach_info_bullet(self, node_from, node_to, delta_t) -> None | Node:
        """
        Simulate trying to reach one node from others.
        :param node_from: origin node
        :param node_to: destination node
        :param delta_t: maximum simulation timesteps
        :return: timesteps to reach node, success indicator (no collision) and new node (very close to 'node_to').
        """

        # Reset env to reset timesteps counter and reset master to track stats
        self.env.reset_master()
        self.env.reset()

        # Place car and target at appropriate poses
        self.env.car.set_pose(node_from.pose)
        self.env.target.set_pose(node_to.pose)

        # Before episode beginning set flags and get initial observation
        done, info, reward = False, False, False
        obs = self.env._get_observation()

        # Loop until termination or delta t elapsed
        while self.env.timestep < delta_t:

            # Get controller's action
            action = self.controller_fn(obs)

            # For RL, we use a lower frequency controller
            if self.cfg.rrt.general.controller == 'rl':

                # Step 'n' times and update t
                for _ in range(self.rl_action_repeat):
                    obs, reward, done, info = self.env.step(action)

                    # Break as soon as done
                    if done:
                        break

            # Non-RL controllers
            else:

                # Step env one time and update t
                obs, reward, done, info = self.env.step(action)

            # Check why episode ended
            if done:
                break

        # If success we set the final pose to be equal to 'node_to' (if config is set to it)
        if not self.cfg.rrt.general.add_last_pose and info['done_info']['success']:
            pose = node_to.pose
        else:
            pose = self.env.car.get_pose()

        # We return a new node avoiding changes in the argument
        node_new = Node(pose, None, None, None, node_to.origin)
        node_new.set_info(NodeReachInfo.new_from_bullet(self.env.timestep, info['done_info'], self.env.stats))

        # Change the origin in case the random node is final but was not reach
        if node_new.origin == 'final' and not info['done_info']['success'] \
                and reward < self.cfg.rrt.general.goal_epsilon:
            node_new.origin = 'final_new'
        
        return node_new

    def _get_reach_info_rs(self, node_from, node_to, delta_t) -> None | Node:
        """
        Get reach info without simulation on pybullet.
        :param node_from: initial node
        :param node_to: destination node
        :param delta_t: maximum simulation timesteps
        :return: info about the trajectory
        """

        # Reset env to reset timesteps counter and reset master to track stats
        self.env.reset_master()
        self.env.reset()

        # Place car and target at appropriate poses
        self.env.car.set_pose(node_from.pose)
        self.env.target.set_pose(node_to.pose)

        # Before episode beginning get initial observation
        obs = self.env._get_observation()

        # Get rs path
        try:
            self.mpcrs._set_rs_path(obs)
        except TypeError:
            return None

        # Loop until termination or delta t elapsed
        reward = -1
        done_info = {}
        rs_idx, bullet_timesteps_elapsed, d = 0, 0, 0
        poses, rewards = [], []
        while bullet_timesteps_elapsed < delta_t:

            # Update current index
            rs_idx += 1
            bullet_timesteps_elapsed = np.round(rs_idx * self.cfg.rrt.rs.step_size / (1/240))

            # Place car at pose
            pose = [self.mpcrs.rs_xs[rs_idx], self.mpcrs.rs_ys[rs_idx], self.mpcrs.rs_yaws[rs_idx]]
            self.env.car.set_pose(pose)

            # Get info from env
            obs = self.env._get_observation()
            reward = self.env.compute_reward(obs['achieved_goal'], obs['desired_goal'])
            done, done_info = self.env._get_done_indicator(reward)

            # Save stats
            poses.append(pose)
            rewards.append(reward)

            # Break episode
            if done:
                break
        
        # If success we set the final pose to be equal to 'node_to'
        if not self.cfg.rrt.general.add_last_pose and done_info['success']:
            pose = node_to.pose
        else:
            pose = self.env.car.get_pose()
        
        # We return a new node avoiding changes in the argument (unless 'node_to' is the final node)
        node_new = Node(pose, None, None, None, node_to.origin)
        node_new.set_info(NodeReachInfo.new_from_rs(self.env.timestep, done_info, poses, rewards))

        # Change the origin in case the random node is final but was not reach (increase threshold)
        if node_new.origin == 'final' and not done_info['success'] \
                and reward < self.cfg.rrt.general.goal_epsilon:
            node_new.origin = 'final_new'
        
        # Get return info
        return node_new

    def _get_sample_final(self) -> Node:
        """
        Simple function to return the final node.
        :return: final node
        """

        return self.node_final

    def _get_sample_sl(self) -> Node:
        """
        Get sample from cvae.
        :return: node from generative model.
        """

        return self.rng.choice(self.list_sl_nodes)

    def _get_sample_uniform(self) -> Node:
        """
        Uniform sample of the goal space.
        :return: random node sample.
        """

        # Get random pose
        low = [0, 0, 0]
        high = [self.cfg.maps.narrow.size, self.cfg.maps.narrow.size, 2 * np.pi]

        # Continue to sample poses until there is no collision
        pose = None
        collided = True
        while collided:

            # Get random pose
            pose = self.rng.uniform(low, high)

            # Place car and check collision
            self.env.car.set_pose(pose)
            collided = self.env._get_collision_indicator()

        # Create node
        return Node(pose, None, None, None, 'random')

    def _get_solution(self) -> list[Node]:
        """
        Get the trajectory going from 'node_init' to 'node_final'.
        :return: list of poses to reach 'node_final' from 'node_init'.
        """

        # Only get trajectory if final node is reachable
        assert self.found_solution, "Final destination not found. Run for more iterations."

        # Start getting the parents from node final
        list_poses = self._get_parents_recursive([self.node_final])
        list_poses.reverse()

        return list_poses

    def _get_solution_cost_length(self):
        """
        Get length of the solution found.
        :return: length of the trajectory
        """

        # Get solution
        list_nodes = self._get_solution()

        # Initialize distance accumulator
        d_total = 0

        # Stats define how the node was reached, so we start at index 1 (initial pose has no stats)
        for node in list_nodes[1:]:
            d_total += node.info.length

        return d_total

    def _get_solution_cost_time(self):
        """
        Get the time of the solution found.
        :return: time of the trajectory
        """

        # Get solution
        list_nodes = self._get_solution()

        # Initialize distance accumulator
        t_total = 0

        # Stats define how the node was reached, so we start at index 1 (initial pose has no stats)
        for node in list_nodes[1:]:
            t_total += node.info.timesteps

        return t_total

    def _get_too_close_indicator(self, cost):
        """
        Check if new node is too close to the nearest node.
        :param cost: number of timesteps for the possible new node.
        :return:
        """

        self._setup_ignore_static_warnings()

        return True if cost < self.min_distance else False

    def _get_within_policy_reach_indicator(self, node_rand):
        """
        Check if policy would be able to reach this pose from any node on the tree,
        :param node_rand: random sampled node
        :return: flag indicating if policy would be able to reach node_rand from node_nearest
        """

        # Don't need this for RS curves
        if self.cfg.rrt.general.distance_fn == 'rs':
            return True

        # We use simple euclidean distance to check if the policy could probably reach the node
        for node in self.list_nodes:
            if euclidean(node.pose[:2], node_rand.pose[:2]) < 4.0:
                return True

        return False

    def _set_add_node(self, node_min, node_new, node_with_info):
        """
        Add new node.
        :param node_min: node that makes new node the closest to root
        :param node_new: new node to be added to the tree
        :param node_with_info: how the 'node_new' was reached from 'node_min'
        """

        # Set the parent / children and update the cost from parent
        node_new.parent = node_min
        node_min.children.append(node_new)

        # Update parent cost and parent reach info
        node_new.cost_from_parent = node_with_info.info.cost
        node_new.set_info(node_with_info.info)

        # Check if cost from root is none first (final pose has none cost from root yet, possibly)
        if node_min.cost_from_root is not None:
            node_new.cost_from_root = node_min.cost_from_root + node_with_info.info.cost

        # If the origin is 'final' the final node was reached with success
        if node_new.origin == 'final':
            self.node_final = node_new
            self.found_solution = True  
        
        # Also add node to the list of all nodes in the tree
        self.list_nodes.append(node_new)

    def _set_init_attributes(self):
        """
        Initialize attributes.
        """

        self.epoch = 0

    def _set_ordinal_model(self):
        """
        Initialize ordinal, scaler and timesteps range.
        """

        # Get the best ordinal config
        cfg, model_path = self._get_ordinal_config_and_path()

        # Initialize best ordinal attributes
        self._set_ordinal_model_init(cfg)

        # Load model
        if self._get_ordinal_model_exists_indicator(model_path):
            io.load(self.ordinal, model_path)

        # Or train model
        else:
            self._set_ordinal_model_training(cfg, model_path)

    def _set_ordinal_model_init(self, cfg: MasterConfig):
        """
        Initialize ordinal and scaler attributes
        :param cfg: the best config for the best ordinal model
        """

        # Initialize dataset
        ds_train = DistanceOrdinalDataset(cfg, 'train', None, None)

        # Initialize model and set attributes
        self.ordinal = SnapshotEnsembleRegressor(Ordinal, cfg.rrt.ordinal.n_estimators, {'cfg': cfg})
        self.ordinal_scaler = ds_train.scaler
        self.ordinal_range = ds_train.timesteps_range

        # Define criterion
        n_classes = int(ds_train.timesteps_range[1] - ds_train.timesteps_range[0] + 1)
        loss = EnsembleLoss(n_classes)

        # Configure model
        self.ordinal.set_criterion(loss)
        self.ordinal.set_optimizer('Adam', lr=1e-3)
        set_logger()

    def _set_ordinal_model_training(self, cfg: MasterConfig, model_path: str):
        """
        Train ordinal model with the best parameters.
        :param cfg: the best config for ordinal model
        """

        # Number of epochs depend on the number of estimators
        epochs = 50
        while epochs % cfg.rrt.ordinal.n_estimators != 0:
            epochs += 1

        # Get data loaders
        ds_train = DistanceOrdinalDataset(self.cfg, 'train', None, None)
        ds_val = DistanceOrdinalDataset(self.cfg, 'val', ds_train.scaler, ds_train.timesteps_range)
        dl_train = DataLoader(ds_train, batch_size=cfg.rrt.ordinal.batch_size, shuffle=True)
        dl_val = DataLoader(ds_val, batch_size=cfg.rrt.ordinal.batch_size)

        # Fit
        self.ordinal.fit(dl_train, test_loader=dl_val, epochs=epochs, save_model=True, save_dir=model_path)

    def _set_rewire(self, list_nodes_near, node_new):
        """
        Rewire step of RRT*.
        :param list_nodes_near: list of nodes near node rand
        :param node_new: new node that was added
        """

        # Loop for all near nodes except the nearest
        for node_near in list_nodes_near:

            # Don't need to test for initial node or 'node_new' parent ('node_min')
            if node_near.origin == 'init' or node_near == node_new.parent:
                continue

            # Now check if using 'node_new' as a waypoint decrease the cost from root
            node_test = self._get_reach_info(node_new, node_near, 2000)

            # None is returned if RS path fails
            if node_test is None:
                continue

            # Debug callback
            self._debug_callback('rewire', locals())

            # If no collision and the node could be reached we compare the costs
            if not node_test.info.done_info['collision'] and node_test.info.done_info['success']:
                
                # Current cost is node near cost from root
                current_cost = node_near.cost_from_root
                
                # New cost is the cost from root of the new node + the travel cost of 'new' to 'near'
                new_cost = node_new.cost_from_root + node_test.info.cost

                # If the new cost is smaller, we rewire
                if new_cost < current_cost:
                    print(f'rewiring (cost decreased) from {current_cost} to {new_cost}...')
                    # Remove 'node_near' from the list of children of the current parent node
                    node_near.set_remove_me_as_child()
                    
                    # We change the parent of the near node to be the new node
                    node_near.parent = node_new
                    node_new.children.append(node_near)
                    
                    # Update the costs and info
                    node_near.cost_from_parent = node_test.info.cost
                    node_near.cost_from_root = new_cost
                    node_near.set_info(node_test.info)
                    
                    # Propagate the change in cost to all children
                    node_near.set_cost_change_propagation()

    def _set_rl_data(self):
        """
        Check if data for ordinal model is present.
        """

        # Check if data exists and RL is loaded
        data_path = Paths().data_rl_distance / self.cfg.rrt.names.rl / 'train.csv'
        if not data_path.exists():
            self.rl.get_distance_data()

    def _test_get_time_to_solve(self):
        """
        Get the time needed to find the solution.
        :return: time needed to solve
        """

        assert self.found_solution, "Solution not found yet..."

        # Read tb event file
        path = next(iter(Paths().exp_tensorboard('rrt', self.cfg.general.config_name_or_prefix).glob("*event*")))
        reader = SummaryReader(log_path=str(path), extra_columns={'wall_time'})
        df = reader.scalars.query("tag == 'found_solution'")

        return df['wall_time'].max() - df['wall_time'].min()

    def _test_get_solution_simulation_cost_length(self):
        """
        Get the time needed to reach the final pose.
        :return: time to reach final pose
        """

        # Should run simulation with controller first
        assert self.solution_simulation_info is not None, "Run test function to set solution simulation info first..."

        return self.solution_simulation_info.ep_stats.get_distance_traveled(0)

    def _test_get_solution_simulation_cost_time(self):
        """
        Get the time needed to reach the final pose.
        :return: time to reach final pose
        """

        # Should run simulation with controller first
        assert self.solution_simulation_info is not None, "Run test function to set solution simulation info first..."

        return self.solution_simulation_info.ep_stats.get_time_to_reach(0)

    def _test_plot_ordinal_accuracy_within_distance(self):
        """
        Plot ordinal model accuracy when there is a single node with minimum distance.
        """

        path = self.path_figs / 'ordAccuracy.png'
        if path.exists():
            return

        # Get test dataset
        ds_test = DistanceOrdinalDataset(self.cfg, 'test', self.ordinal_scaler, self.ordinal_range)

        # Predict distances
        y_pred_all = corn_label_from_logits(self.ordinal.predict(ds_test.x)).detach().cpu().flatten()

        # Get maximum possible value
        y_max = ds_test.y.max()

        # Data struct
        df = pd.DataFrame(columns=['n', 'x', 'success_rate'])

        # Loop x values
        xs = [6, 8, 10]
        for x in xs:

            # Loop n values
            # ns = range(x + 1, 500 + x + 2, 50)
            ns = range(100, 550, 50)
            for n in ns:

                # Check success rate for this number of nodes
                success_counter = 0
                for t in range(100):

                    # Get min value
                    min_value_lower_than_max = False
                    y_min_idx, y_min_value = None, None
                    while not min_value_lower_than_max:
                        y_min_idx = np.random.randint(0, y_pred_all.shape[0], 1)[0]
                        y_min_value = ds_test.y[y_min_idx]
                        min_value_lower_than_max = y_min_value < y_max

                    # Get list of idxs of predictions
                    idxs = [y_min_idx]
                    while len(idxs) < n:
                        idx = np.random.randint(0, y_pred_all.shape[0], 1)[0]
                        val = ds_test.y[idx]
                        if val > y_min_value:
                            idxs.append(idx)

                    # Get values at chosen indexes
                    y_true = ds_test.y[np.array(idxs)]
                    y_pred = y_pred_all[np.array(idxs)]

                    # Sorted indexes
                    idxs_true_sorted = np.array(np.argsort(y_true))
                    idxs_pred_sorted = np.array(np.argsort(y_pred))

                    # Find index where prediction is 'x' greater than minimum prediction
                    min_pred = y_pred[idxs_pred_sorted[0]]
                    stop_at = 1
                    for i, pred in enumerate(y_pred[idxs_pred_sorted]):
                        if np.abs(pred - min_pred) > x:
                            stop_at = i
                            break

                    print(stop_at)

                    # Check if top-x from y_pred matches top-1 of true
                    if np.any(idxs_pred_sorted[:stop_at] == 0):
                        success_counter += 1

                    # Check if argmin == y_min_index (zero)
                    # if np.argmin(y_pred) == 0:
                    #     success_counter += 1

                # Add to df
                row = pd.DataFrame(dict(n=[n], x=[x], success_rate=[success_counter]))
                df = df.append(row, ignore_index=True)

        # Change formatting
        df['x'] = df['x'].apply(lambda x: str(x) + '%')

        # Rename df columns before plotting
        df.rename(columns={'x': 'Simulações', 'success_rate': 'Taxa de sucesso'}, inplace=True)

        # Plot
        plt.close()
        sns.set_style('whitegrid')
        line_plot = sns.lineplot(x="n", y="Taxa de sucesso", hue="Simulações", size="Simulações",
                                 style="Simulações",
                                 sizes=(3, 1), data=df)

        # Save
        fig = line_plot.get_figure()
        fig.savefig(path, bbox_inches="tight", dpi=600)

    def _test_plot_ordinal_accuracy_top_x_unique_min(self):
        """
        Plot ordinal model accuracy when there is a single node with minimum distance.
        """

        path = self.path_figs / 'ordAccuracyTopXPercentUniqueMin.png'
        if path.exists():
            return

        # Get test dataset
        ds_test = DistanceOrdinalDataset(self.cfg, 'test', self.ordinal_scaler, self.ordinal_range)

        # Predict distances
        y_pred_all = corn_label_from_logits(self.ordinal.predict(ds_test.x)).detach().cpu().flatten()

        # Get maximum possible value
        y_max = ds_test.y.max()

        # Data struct
        df = pd.DataFrame(columns=['n', 'x', 'success_rate'])

        # Loop x values
        xs = [10, 20, 30, 40, 50]
        for x in xs:

            # Loop n values
            # ns = range(x + 1, 500 + x + 2, 50)
            ns = range(100, 550, 50)
            for n in ns:

                # Get number of x to test
                nx = round(0.01 * x * n)

                # Check success rate for this number of nodes
                success_counter = 0
                for t in range(100):

                    # Get min value
                    min_value_lower_than_max = False
                    y_min_idx, y_min_value = None, None
                    while not min_value_lower_than_max:
                        y_min_idx = np.random.randint(0, y_pred_all.shape[0], 1)[0]
                        y_min_value = ds_test.y[y_min_idx]
                        min_value_lower_than_max = y_min_value < y_max

                    # Get list of idxs of predictions
                    idxs = [y_min_idx]
                    while len(idxs) < n:
                        idx = np.random.randint(0, y_pred_all.shape[0], 1)[0]
                        val = ds_test.y[idx]
                        if val > y_min_value:
                            idxs.append(idx)

                    # Get values at chosen indexes
                    y_true = ds_test.y[np.array(idxs)]
                    y_pred = y_pred_all[np.array(idxs)]

                    # Sorted indexes
                    idxs_true_sorted = np.array(np.argsort(y_true))
                    idxs_pred_sorted = np.array(np.argsort(y_pred))

                    # Check if top-x from y_pred matches top-1 of true
                    if np.any(idxs_pred_sorted[:nx] == 0):
                        success_counter += 1

                    # Check if argmin == y_min_index (zero)
                    # if np.argmin(y_pred) == 0:
                    #     success_counter += 1

                # Add to df
                row = pd.DataFrame(dict(n=[n], x=[x], success_rate=[success_counter]))
                df = df.append(row, ignore_index=True)

        # Change formatting
        df['x'] = df['x'].apply(lambda x: str(x) + '%')

        # Rename df columns before plotting
        df.rename(columns={'x': 'Simulações', 'success_rate': 'Taxa de sucesso'}, inplace=True)

        # Plot
        plt.close()
        sns.set_style('whitegrid')
        line_plot = sns.lineplot(x="n", y="Taxa de sucesso", hue="Simulações", size="Simulações", style="Simulações",
                                 sizes=(3, 1), data=df)

        # Save
        fig = line_plot.get_figure()
        fig.savefig(path, bbox_inches="tight", dpi=600)

    def _test_plot_ordinal_accuracy_unique_min(self):
        """
        Plot ordinal model top-x% accuracy when there is a single node with minimum distance.
        :return:
        """

        path = self.path_figs / 'ordAccuracyUniqueMin.png'
        if path.exists():
            return

        # Get test dataset
        ds_test = DistanceOrdinalDataset(self.cfg, 'test', self.ordinal_scaler, self.ordinal_range)

        # Predict distances
        y_pred_all = corn_label_from_logits(self.ordinal.predict(ds_test.x)).detach().cpu().flatten()

        # Get maximum possible value
        y_max = ds_test.y.max()

        # Data struct
        df = pd.DataFrame(columns=['n', 'x', 'success_rate'])

        ns = range(10, 510, 10)
        for n in ns:

            # Check success rate for this number of nodes
            success_counter = 0
            for t in range(100):

                # Get min value
                min_value_lower_than_max = False
                y_min_idx, y_min_value = None, None
                while not min_value_lower_than_max:
                    y_min_idx = np.random.randint(0, y_pred_all.shape[0], 1)[0]
                    y_min_value = ds_test.y[y_min_idx]
                    min_value_lower_than_max = y_min_value < y_max

                # Get list of idxs of predictions
                idxs = [y_min_idx]
                while len(idxs) < n:
                    idx = np.random.randint(0, y_pred_all.shape[0], 1)[0]
                    val = ds_test.y[idx]
                    if val > y_min_value:
                        idxs.append(idx)

                # Get values at chosen indexes
                y_pred = y_pred_all[np.array(idxs)]

                # Sorted indexes
                idxs_pred_sorted = np.array(np.argsort(y_pred))

                # Check if top-x from y_pred matches top-1 of true
                if idxs_pred_sorted[0] == 0:
                    success_counter += 1

            # Add to df
            row = pd.DataFrame(dict(n=[n], success_rate=[success_counter]))
            df = df.append(row, ignore_index=True)

        # Rename df columns before plotting
        df.rename(columns={'success_rate': 'Taxa de sucesso'}, inplace=True)

        # Plot
        plt.close()
        sns.set_style('whitegrid')
        line_plot = sns.lineplot(x="n", y="Taxa de sucesso", data=df)

        # Save
        fig = line_plot.get_figure()
        fig.savefig(path, bbox_inches="tight", dpi=600)

    def _test_plot_solution_top_view(self):
        """
        Plot top view of solution trajectory with simulation.
        """

        # Return if figure exists
        path = self.path_figs / 'rrtSolutionTopView.png'
        if path.exists():
            return

        # Should run simulation with controller first
        assert self.solution_simulation_info is not None, "Run test function to set solution simulation info first..."

        # Place ghosts 
        self.env._set_place_trajectory_ghosts(0, self.solution_simulation_info.ep_stats)

        # Take screenshot
        img = self.env._get_image_current_top_view()

        # Plot 
        fig, ax = plt.subplots()
        ax.imshow(img, origin='lower', extent=self.cfg.env.test.narrow.img_extent)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.savefig(path, dpi=600, bbox_inches='tight')

    def _test_plot_solution_tree(self):
        """
        Plot RRT tree with a different color for the solution.
        """

        # Return if figure exists
        path = self.path_figs / 'rrtSolutionTree.png'
        if path.exists():
            return

        # Plot tree normally
        mmap = Map(self.cfg)
        mmap.set_add_nodes(self.list_nodes)
        mmap.set_add_nodes(self.node_final)
        fig, ax = mmap.get_plot_tree()

        # Now we override branches for the solution (different color and no transparency)
        mmap._set_plot_branches_for_list(ax, self._get_solution())

        # Save
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        fig.savefig(path, dpi=600, bbox_inches='tight')

    def _setup(self, *args, **kwargs):
        """
        Set up.
        """

        # Set up base algorithm settings
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

        # Set up rrt settings
        self._setup_seed()
        self._setup_cost_type()
        self._setup_conditions()
        self._setup_nodes_init_final()
        self._setup_obstacles()
        self._setup_list_nodes()
        self._setup_env()
        self._setup_sl()
        self._setup_controller_distance_fns()
        self._setup_controller_model()
        self._setup_distance_model()
        self._setup_debug()

    def _setup_attrs_to_save(self):
        """
        Set up attributes to save on checkpoint.
        """

        self.save_attrs = ['epoch', 'model_curr_loss', 'model_best_loss', 'training_over', 'found_solution',
                           'list_nodes', 'list_sl_nodes', 'node_final', 'node_init', 'narrow1_pos', 'narrow2_pos',
                           'ordinal_scaler', 'ordinal_range', 'cutoff', 'min_distance']

    def _setup_checkpoint(self):
        """
        Set up checkpoint.
        """

        # Check if there is a checkpoint
        self.load_checkpoint = self._get_checkpoint_exists_indicator()

        # Load checkpoint
        if self.load_checkpoint:

            # Load attributes
            self._set_load_attributes()

            # Load test checkpoint
            path = Paths().exp_checkpoint('rrt', self.cfg.general.config_name_or_prefix) / 'simulation_info'
            path_part = path.parent / 'simulation_info01'
            if path_part.exists():
                self.solution_simulation_info = RRTSolutionInfo.load_from_file(path)

    def _setup_conditions(self):
        """
        Setup initial (root) node and final (goal) nodes.
        """

        # Load config from experiment name
        cfg = MasterConfig.load_from_experiment('sl', self.cfg.rrt.names.sl)
        
        # If condition number is int we get from validation set
        cond = self.cfg.rrt.general.condition
        if isinstance(cond, int):
            obstacles, state_i, state_f = SL(cfg).get_conditions(cond)
            self.cfg.maps.narrow.narrow1_pos = obstacles[0:2]
            self.cfg.maps.narrow.narrow2_pos = obstacles[2:4]
            self.cfg.env.car.pose = state4_to_pose3(state_i)
            self.cfg.env.target.pose = state4_to_pose3(state_f)

        # Or generate random values
        elif isinstance(cond, str):
            if cond == 'random':
                self.cfg.maps.narrow.narrow1_pos = np.round(self.rng.uniform([3, 2], [4, 7]), 1)
                self.cfg.maps.narrow.narrow2_pos = np.round(self.rng.uniform([6, 2], [7, 7]), 1)
                self.cfg.env.car.pose = np.round(self.rng.uniform([0.8, 2.0, 0], [1.2, 7.0, 2*np.pi]), 1)
                self.cfg.env.target.pose = np.round(self.rng.uniform([8.8, 2.0, 0], [9.2, 9.0, 2*np.pi]), 1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def _setup_controller_distance_fns(self):
        """
        Set up controller and distance functions according to config.
        """

        # Set controller function
        match self.cfg.rrt.general.controller:
            case "rl":
                self.controller_fn = self._get_action_rl
            case "mpc":
                self.controller_fn = self._get_action_mpc
            case _:
                raise NotImplementedError

        # Set distance function between goals functions
        match self.cfg.rrt.general.distance_fn:
            case _ if "rl-ordinal" in self.cfg.rrt.general.distance_fn:
                self.distance_fn = self._get_distance_ordinal
            case "rl-sim":
                self.distance_fn = self._get_distance_sim
            case "rs":
                self.distance_fn = self._get_distance_rs
            case _:
                raise NotImplementedError

    def _setup_controller_model(self):
        """
        Set up agent for RL control.
        """

        # Only RL controller has a model
        if self.cfg.rrt.general.controller != "rl":
            return

        # Load RL
        rl_cfg = MasterConfig.load_from_experiment('rl', self.cfg.rrt.names.rl)
        self.rl = RL(rl_cfg)

        # Set action repeat
        self.rl_action_repeat = int(1 / (rl_cfg.env.car.f_val * (1 / 240)))

    def _setup_cost_type(self):
        """
        Cutoff is different depending on the distance function.
        """

        # Don't need if loaded checkpoint
        if self.load_checkpoint:
            return
        
        match self.cfg.rrt.general.distance_fn:
            case 'rs':
                self.cutoff = self.cfg.rrt.near.cutoff_length
                self.min_distance = self.cfg.rrt.nearest.min_length
            case _:
                self.cutoff = self.cfg.rrt.near.cutoff_timesteps
                self.min_distance = self.cfg.rrt.nearest.min_n_timesteps

    def _setup_debug(self):
        """
        Debug algorithm with graphs.
        """

        if self.cfg.rrt.general.debug:
            
            # Create map with present nodes
            self.dbg_map = Map(self.cfg)
            self.dbg_map.set_add_nodes(self.list_nodes)
            self.dbg_map.set_add_nodes(self.node_final)
            
            # Initialize figure
            self.dbg_fig, self.dbg_ax = self.dbg_map.get_plot_lines()
            
            # Set legend
            # self.dbg_ax.scatter(-1, -1, color='green', label='initial node')
            # self.dbg_ax.scatter(-1, -1, color='red', label='final node')
            # self.dbg_ax.scatter(-1, -1, color='lime', label='new node')
            # self.dbg_ax.scatter(-1, -1, color='purple', label='min cost node')
            # self.dbg_ax.scatter(-1, -1, color='aqua', label='nearest node')
            # self.dbg_ax.scatter(-1, -1, color='blue', label='random node')
            # self.dbg_ax.scatter(-1, -1, color='chocolate', label='near node')
            # self.dbg_ax.legend(bbox_to_anchor=(1,1), loc="upper left")
            
            # Show
            self.dbg_fig.show()
        
        elif not self.cfg.general.is_train:
            return

        else:
            
            # Non-interactive backend
            mpl.use('agg')

    def _setup_distance_model(self):
        """
        Set up distance prediction model. This is needed for ordinal model.
        """

        # For Redd-Shepps we initialize the class and for RL we train ordinal model
        match self.cfg.rrt.general.distance_fn:
            case 'rs':
                self.mpcrs = MPCRS(self.cfg)
            case _ if 'rl-ordinal' in self.cfg.rrt.general.distance_fn:
                self._set_rl_data()
                self._set_ordinal_model()

    def _setup_env(self):
        """
        Set up PyBullet simulation.
        """

        self.cfg.env.general.gui = True if not self.cfg.general.is_train else False
        self.cfg.env.reward.epsilon *= self.cfg.rrt.general.epsilon_multiplier
        self.env = CarNavigationBulletEnv(self.cfg)

    def _setup_ignore_static_warnings(self):
        """
        Ignore static method warnings.
        """

        pass

    def _setup_init_base_attrs(self):
        """
        Initialize base attributes (epoch, model losses, etc)
        """

        # Base attributes
        self.epoch = 0
        self.model_curr_loss = np.inf
        self.model_best_loss = np.inf
        self.training_over = False
        
        # RRT attributes
        self.found_solution = False

    def _setup_list_nodes(self):
        """
        Set up list of nodes in the tree.
        """

        # Don't need if loaded checkpoint
        if self.load_checkpoint:
            return

        # Initialize graph structure
        self.list_nodes = [self.node_init]

    def _setup_nodes_init_final(self):
        """
        Set up initial and final nodes.
        """

        if self.load_checkpoint:
            return
        
        self.node_init = Node(self.cfg.env.car.pose, None, 0, 0, 'init')
        self.node_final = Node(self.cfg.env.target.pose, None, None, None, 'final')

    def _setup_obstacles(self):
        """
        Save obstacles to attribute.
        """

        if self.load_checkpoint:
            return

        self.narrow1_pos = self.cfg.maps.narrow.narrow1_pos
        self.narrow2_pos = self.cfg.maps.narrow.narrow2_pos

    def _setup_seed(self):
        """
        Set up seed for random processes. Must be called before setup SL because it is used to generate states.
        """

        if self.cfg.rrt.general.seed is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.cfg.rrt.general.seed)

    def _setup_sl(self):
        """
        Set up cvae.
        """

        # Don't need if loading checkpoint
        if self.load_checkpoint:
            return

        # Load algorithm from name
        sl_cfg = MasterConfig.load_from_experiment('sl', self.cfg.rrt.names.sl)
        sl = SL(sl_cfg)

        # Initialize conditions vector
        y = []

        # Add obstacles
        x_o_1, y_o_1 = self.cfg.maps.narrow.narrow1_pos
        x_o_2, y_o_2 = self.cfg.maps.narrow.narrow2_pos
        y.extend([x_o_1, y_o_1, x_o_2, y_o_2])

        # Add initial and final pose
        y.extend(pose3_to_state4_sl(self.cfg.env.car.pose))
        y.extend(pose3_to_state4_sl(self.cfg.env.target.pose))

        # Generate states with CVAE model
        states = sl.get_samples(np.array(y), 2000, self.rng)
        states = state4_sl_to_state4_rl(states)

        # Generate list of nodes from these states
        self.list_sl_nodes = []
        for st in states:
            self.list_sl_nodes.append(Node(state4_to_pose3(st), None, None, None, "sl"))

    def _setup_stats(self):
        """
        Set up stats object.
        """

        if self.load_checkpoint:
            self.stats = RRTStats.load_from_file(self.path_stats)
        else:
            self.stats = RRTStats.new(self.cfg)
            self.stats.save_to_file(self.path_stats)

    @staticmethod
    def _objective(trial: optuna.Trial, kwargs: dict):
        """
        Objective for optuna.
        :param trial: trial argument
        :return: objective evaluation
        """

        # Get from config
        batch_size = kwargs['cfg'].rrt.ordinal.opt_batch_size
        n_layers = kwargs['cfg'].rrt.ordinal.opt_n_layers
        n_units = kwargs['cfg'].rrt.ordinal.opt_n_units
        n_estimators = kwargs['cfg'].rrt.ordinal.opt_n_estimators

        # Suggest architecture
        n_layers = trial.suggest_int('n_layers', n_layers[0], n_layers[1])
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_{i}', n_units[0], n_units[1], step=n_units[2]))

        # Suggest batch size
        batch_size = trial.suggest_int('batch_size', batch_size[0], batch_size[1], step=batch_size[2])

        # Suggest number of estimators
        n_estimators = trial.suggest_int('n_estimators', n_estimators[0], n_estimators[1], step=n_estimators[2])

        # Get data loaders
        dl_train = DataLoader(kwargs['ds_train'], batch_size=batch_size, shuffle=True)
        dl_val = DataLoader(kwargs['ds_val'], batch_size=batch_size)

        # Number of epochs depend on the number of estimators
        epochs = 50
        while epochs % n_estimators != 0:
            epochs += 1

        # Define criterion
        n_classes = int(kwargs['ds_train'].timesteps_range[1] - kwargs['ds_train'].timesteps_range[0] + 1)
        loss = EnsembleLoss(n_classes)

        # Initialize model
        model = SnapshotEnsembleRegressor(estimator=Ordinal, n_estimators=n_estimators, cuda=True,
                                          estimator_args={'cfg': kwargs['cfg']})

        # Configure model
        model.set_criterion(loss)
        model.set_optimizer('Adam', lr=1e-3)
        set_logger()

        # Create unique name based on the current trial
        trial_num = 0
        path_base = Paths().exp('rrt', kwargs['cfg'].general.config_name_or_prefix) / 'checkpoint' / 'param_search'
        path_base = str(path_base / 'model_')
        path_full = path_base + str(trial_num).zfill(2)
        while Path(path_full).exists():
            trial_num += 1
            path_full = path_base + str(trial_num).zfill(2)

        # Fit model
        model.fit(train_loader=dl_train, epochs=epochs, test_loader=dl_val, save_dir=path_full)

        # Get validation error
        logits = model.predict(kwargs['ds_val'].x)
        y_pred = corn_label_from_logits(logits)
        mae = np.mean(np.abs(kwargs['ds_val'].y - y_pred.detach().numpy().flatten()))

        return mae

    def _debug_callback(self, mode: str, loc):
        """
        Debug callback.
        """

        # Check if it is debugging
        if self.cfg.rrt.general.debug:

            # Clear previous phase
            self.dbg_map.set_clear_temp_objs()

            # Each debugging phase is a case
            match mode:

                # Plot random node, distances from that node and nearest node
                case 'nearest':

                    # Get locals
                    node_rand = loc['node_rand']
                    distances = loc['distances']
                    nearest_idx = loc['nearest_idx']

                    # Add random node and nearest node
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, node_rand, 'dodgerblue')
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, self.list_nodes[nearest_idx], 'aqua')

                    # Add distances
                    for i, n in enumerate(self.list_nodes):
                        self.dbg_map.set_add_temp_text(self.dbg_ax, n.pose, f'{distances[i]:.2f}', 'black')

                # Plot new node and trajectory from nearest to this new node
                case 'new':

                    # Get locals
                    node_rand = loc['node_rand']
                    node_nearest = loc['node_nearest']
                    node_new = loc['node_new']

                    # Add random node, nearest node and new node
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, node_rand, 'dodgerblue')
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, node_nearest, 'aqua')
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, node_new, 'lime')

                    # Add trajectory from nearest to new
                    self.dbg_map.set_add_temp_line(self.dbg_ax, node_new, 'lime')

                # Plot new node and nodes near (with distance)
                case 'near':
                    
                    # Get locals
                    node_new = loc['node_new']
                    list_nodes_near = loc['list_nodes_near']
                    distances_near = loc['distances_near']
                    
                    # Add new node
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, node_new, 'lime')
                    
                    # Add near nodes
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, list_nodes_near, 'brown')

                    # Add distances
                    for i, n in enumerate(list_nodes_near):
                        self.dbg_map.set_add_temp_text(self.dbg_ax, n.pose,
                                                       f'distance (new):{distances_near[i]:.2f}', color='brown')

                # Plot new node, near nodes and the cost of new node for each near node
                case 'min':

                    # Get locals
                    node_new = loc['node_new']
                    list_nodes_near = loc['list_nodes_near']
                    costs = loc['costs']

                    # Add new node
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, node_new, 'lime')

                    # Add near nodes
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, list_nodes_near, 'brown')

                    # Now plot the cost from root of new node for each near node
                    for n, c in zip(list_nodes_near, costs):
                        self.dbg_map.set_add_temp_text(self.dbg_ax, n.pose,
                                                       f'cost (root): {c:.2f}', 'brown')

                # Add new node to the tree
                case 'add':

                    # Get locals
                    node_min = loc['node_min']
                    node_new = loc['node_new']
                    node_with_info = loc['node_with_info']

                    # Plot min node and new node
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, node_min, 'orange')
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, node_new, 'lime')

                    # Plot line between min and new
                    self.dbg_map.set_add_temp_line(self.dbg_ax, node_with_info, color='orange')

                # Plot rewiring for each near node
                case 'rewire':

                    # Get locals
                    node_new = loc['node_new']
                    node_near = loc['node_near']
                    node_test = loc['node_test']

                    # Add new node and this node's cost from root
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, node_new, 'lime')
                    self.dbg_map.set_add_temp_text(self.dbg_ax, node_new.pose,
                                                   f'cost (root): {node_new.cost_from_root:.2f}', 'lime')

                    # Add near node
                    self.dbg_map.set_add_temp_arrows(self.dbg_ax, node_near, 'purple')

                    # Add cost from new and current cost
                    self.dbg_map.set_add_temp_text(self.dbg_ax, node_near.pose,
                                                   f'cost (root): {node_near.cost_from_root:.2f}', 'purple')
                    self.dbg_map.set_add_temp_text(self.dbg_ax, [node_near.pose[0], node_near.pose[1] - 0.2],
                                                   f'cost (parent): {node_test.info.cost:.2f}', 'lime')

                # Update list of nodes of the base map
                case 'update':
                    
                    # Clear axis
                    self.dbg_ax.clear()
                    
                    # Create map with current nodes
                    self.dbg_map = Map(self.cfg)
                    self.dbg_map.set_add_nodes(self.list_nodes)
                    self.dbg_map.set_add_nodes(self.node_final)
                    
                    # Draw on axis
                    self.dbg_ax = self.dbg_map.get_plot_lines(self.dbg_ax)
                    
                    # Plot costs from root
                    for node in self.list_nodes:
                        self.dbg_map.set_add_temp_text(
                            self.dbg_ax, node.pose, f'{node.cost_from_root:.2f}', 'red'
                        )
