import time
from ctypes import windll

import gym
import pybullet as p
import pybullet_data as pd
from PIL import Image
from numpy.random import Generator

from rrt_ml.utilities.configs import *
from rrt_ml.utilities.formulas import *
from rrt_ml.utilities.hints import *
from rrt_ml.utilities.maps import *
from rrt_ml.utilities.misc import *
from rrt_ml.utilities.paths import *
from rrt_ml.utilities.stats import *


class CarNavigationBulletEnv(gym.Env):

    def __init__(self, cfg: MasterConfig):
        """
        Initialize env.
        """

        self.cfg = cfg  # type: MasterConfig | None
        self.reward_weights = None  # type: Vector2 | None
        self.action_repeat = None  # type: int | None
        self.car = None  # type: Car | None
        self.target = None  # type: Target | None
        self.obstacles = None  # type: list[Obstacle, ...] | None
        self.init_pose = None  # type: Vector3 | None
        self.final_pose = None  # type: Vector3 | None
        self.timestep = None  # type: int | None
        self.bullet_timestep = None  # type: int | None
        self.episode_num = None  # type: int | None
        self.l_ghost_ids = None  # type: list[list[int, ...], ...] | None
        self.save_state = None  # type: int | None
        self.stats = None  # type: EnvStats | None
        self.rng = None  # type: Generator | None
        self.img_crop = None  # type: Vector4 | None
        self.img_extent = None  # type: Vector4 | None

        self._setup()

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: dict = None):
        """
        Common API for calculating rewards in goal conditioned tasks.
        :param achieved_goal: goal achieved
        :param desired_goal: behavioural goal
        :param info: additional info
        :return: calculated reward
        """

        self._setup_ignore_static_warnings(info)

        # Calculate p-norm reward (works for vectorized calculations)
        abs_dif = np.abs(achieved_goal - desired_goal)
        weighted_abs_dif = np.dot(abs_dif, self.reward_weights)
        reward = -np.power(weighted_abs_dif, self.cfg.env.reward.p_val)

        # Make reward sparse (0 or 1 only)
        reward = self._get_reward_sparse(reward)

        return reward

    def render(self, mode="human") -> None:
        """
        Common API.
        """

        self._setup_ignore_static_warnings()
        print(f'Create env with "GUI" set to "True". Mode: {mode}')

    def reset(self, **kwargs):
        """
        Reset env.
        :param kwargs: k/v 'reset_seed'/bool to set the seed
        :return: initial observation
        """

        self._set_restore_state()
        self._set_place()
        self._set_reset_joints()
        self._set_increment_episode_num()
        self._set_zero_timesteps()
        self._set_stats()

        return self._get_observation()

    def reset_master(self):
        """
        Reset for eval env to track trajectories with deterministic resetting.
        """

        self._set_zero_episode_num()
        self._set_seed()

    def step(self, action: Vector2) -> tuple[dict, float, bool, dict]:
        """
        Step environment.
        :param action: agent action
        :return: tuple [next_obs, reward, done, info]
        """

        # Get current observation to save
        obs = self._get_observation()

        # Scale action from [-1, 1] to [v/phi_min v/phi_max] and set motors refs
        v_ref = scale_to_range(action[0], [-1, 1], [-self.cfg.env.car.v_max, +self.cfg.env.car.v_max])
        phi_ref = scale_to_range(action[1], [-1, 1], [-self.cfg.env.car.phi_max, +self.cfg.env.car.phi_max])
        self.car.set_motors_refs_ackermann(v_ref, phi_ref)

        # Action according to frequency
        for i in range(self.action_repeat):

            # Record stats
            if self.cfg.env.general.stats:

                # Normal stats
                self._set_bullet_stats()
                self._set_bullet_timestep()

            # Step bullet
            p.stepSimulation(physicsClientId=self.client)

        # Calculate reward
        next_obs = self._get_observation()
        reward = self.compute_reward(next_obs['achieved_goal'], next_obs['desired_goal'], {})
        done, done_info = self._get_done_indicator(reward)

        # Update if in eval mode
        if self.cfg.env.general.stats:
            # Update mdp stats
            self._set_mdp_stats(Transition.new(obs, next_obs, action, reward, done))

            # Get top view if done
            # self._set_new_image_top_view()

        # Update timestep
        self._set_increment_mdp_timestep()

        return next_obs, reward, done, {'stats': self.stats, 'done_info': done_info}

    def seed(self, num):
        """
        Do nothing.
        """
        
        return

    def _get_collision_indicator(self):
        """
        Check if the car collided with any obstacle.
        :return: boolean indicator of collision
        """

        # Return true on the first contact
        if self.obstacles is not None:
            for obstacle in self.obstacles:
                contact = (
                        len(
                            p.getContactPoints(
                                self.car.id, obstacle.id, physicsClientId=self.client
                            )
                        )
                        > 0
                )
                if contact:
                    return True

        return False

    def _get_done_indicator(self, reward: float) -> tuple[bool, dict[str, bool]]:
        """
        Check if episode should end.
        :return: boolean indicator and a dict with reasons to terminate (whether they eval as True or False)
        """

        # Get all boolean indicators
        collided = self._get_collision_indicator()
        time_is_up = self._get_time_is_up_indicator()
        goal_reached = self._get_goal_reached_indicator(reward)

        # Get info
        d = dict(collision=collided, time=time_is_up, success=goal_reached)

        return collided or time_is_up or goal_reached, d

    def _get_goal_reached_indicator(self, reward: float) -> bool:
        """
        Check if goal is reached by checking if 'reward' > 'epsilon' (tolerance).
        :param reward: reward at this time step
        :return: boolean indicating if goal was reached
        """

        return reward > self.cfg.env.reward.epsilon

    def _get_image_current_top_view(self) -> Image:
        """
        Get top view image from bullet renderer.
        :return: PIL image
        """

        try:
            import win32gui
            import win32ui
        except ImportError:
            return

        # Get image
        hwnd = win32gui.FindWindow('DeviceWin32', None)
        left, top, right, bot = win32gui.GetClientRect(hwnd)
        windll.user32.SetProcessDPIAware()
        w = right - left
        h = bot - top
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        save_bit_map = win32ui.CreateBitmap()
        save_bit_map.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(save_bit_map)
        windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 2)
        bmp_info = save_bit_map.GetInfo()
        bmp_str = save_bit_map.GetBitmapBits(True)
        img = Image.frombuffer('RGB', (bmp_info['bmWidth'], bmp_info['bmHeight']), bmp_str, 'raw', 'BGRX', 0, 1)

        # Cleanup
        win32gui.DeleteObject(save_bit_map.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)

        # Adjust image
        new_img = img.crop(self.img_crop)
        new_img = new_img.transpose(Image.FLIP_TOP_BOTTOM)

        return new_img

    def _get_observation(self):
        """
        Get env observation.
        :return: dict observation
        """

        # Get values
        state = self.car.get_state()
        achieved_goal = self.car.get_achieved_goal()
        desired_goal = self.target.get_desired_goal()
        mpc_state = self.car.get_state_mpc()

        return dict(
            observation=state, achieved_goal=achieved_goal, desired_goal=desired_goal, mpc_state=mpc_state
        )

    def _get_reward_sparse(self, reward: Vector):
        """
        Make reward sparse (0 or -1 values only).
        :param reward: single reward or array
        :return: sparse reward
        """

        # Vectorized calculation
        if isinstance(reward, list | tuple | np.ndarray):
            for i, r in enumerate(reward):
                if r < self.cfg.env.reward.epsilon:
                    reward[i] = -1
                else:
                    reward[i] = 0
                    # reward[i] = r

        # Single reward
        else:
            if reward < self.cfg.env.reward.epsilon:
                reward = -1
            else:
                reward = 0
                # reward = reward

        return reward

    def _get_time_is_up_indicator(self):
        """
        Check end of episode by number of timesteps.
        :return: boolean indicating that maximum timesteps was reached
        """

        return self.timestep >= (self.cfg.env.general.max_timestep - 1)

    def _set_bullet_stats(self):
        """
        Add bullet physics statistics after each bullet step call.
        """

        # Alias
        ep = self.episode_num
        t = self.bullet_timestep

        # Add basic info
        self.stats.bullet.time[ep, t, ...] = t * 1 / 240
        self.stats.bullet.car_poses[ep, t, ...] = self.car.get_pose()
        self.stats.bullet.target_poses[ep, t, ...] = self.target.get_pose()

        # Calculate reward and add 'done'
        obs = self._get_observation()
        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], {})
        self.stats.bullet.rewards[ep, t, ...] = reward
        self.stats.bullet.dones[ep, t, ...] = self._get_done_indicator(reward)

        # Add steer positions
        steer_pos1, _, _, _ = p.getJointState(self.car.id, 4, physicsClientId=self.client)
        steer_pos2, _, _, _ = p.getJointState(self.car.id, 6, physicsClientId=self.client)
        self.stats.bullet.car_steers[ep, t, ...] = [steer_pos1, steer_pos2]

        # Check velocity magnitude and then signal
        v, _ = p.getBaseVelocity(self.car.id, physicsClientId=self.client)
        pos, vel, _, _ = p.getJointState(self.car.id, 2, physicsClientId=self.client)
        sign = 1 if vel > 0 else -1
        self.stats.bullet.car_velocities[ep, t, ...] = sign * np.linalg.norm(v[:2])

        # Add wall time
        self.stats.bullet.wall_time[ep, t, ...] = time.time()

    def _set_bullet_timestep(self):
        """
        Increment bullet timestep counter.
        """

        self.bullet_timestep += 1

    def _set_place_trajectory_ghosts(self, episode_num: int, stats: EnvStats | None = None):
        """
        Place ghosts in trajectory to take screenshot.
        :param episode_num: number of the eval episode
        :param stats: current env stats or other stats object
        """

        # Get the stats where poses come from
        if stats is None:
            stats = self.stats

        # Indexes = [start, end + step, step]
        interval = self.cfg.env.test.general.ghost_interval
        n_timesteps = len(stats.bullet.car_poses[0, :, 0])
        idxs = np.arange(0, n_timesteps, interval)

        # Get poses and steering angles
        poses = stats.bullet.car_poses[episode_num, idxs, ...]
        steers = stats.bullet.car_steers[episode_num, idxs, ...]

        # Remove zeros
        poses = remove_trailing_zeros(poses).reshape(-1, 3)
        steers = remove_trailing_zeros(steers).reshape(-1, 2)

        self.l_ghost_ids = []
        for pose, steer in zip(poses, steers):
            self._set_place_ghost_car(pose, steer)

    def _set_remove_ghosts(self):
        """
        Remove trajectory ghosts from GUI.
        """

        # Return if no ghosts on scene
        if self.l_ghost_ids is None:
            return
        if isinstance(self.l_ghost_ids, list):
            if len(self.l_ghost_ids) == 0:
                return

        # Remove bodies
        for idd in self.l_ghost_ids:
            p.removeBody(idd, physicsClientId=self.client)

    def _set_increment_episode_num(self):
        """
        Increment number of episodes.
        """

        self.episode_num += 1

    def _set_increment_mdp_timestep(self):
        """
        Set simulation timestep value or increase current value by one.
        """

        self.timestep += 1 if self.timestep is not None else 0

    def _set_mdp_stats(self, transition: 'Transition'):
        """
        Add current time step info if on test mode.
        """

        # Alias
        ep = self.episode_num
        t = self.timestep

        # Add
        self.stats.mdp.states[ep, t, ...] = transition.state
        self.stats.mdp.achieved_goals[ep, t, ...] = transition.achieved_goal
        self.stats.mdp.desired_goals[ep, t, ...] = transition.desired_goal
        self.stats.mdp.next_states[ep, t, ...] = transition.next_state
        self.stats.mdp.next_achieved_goals[ep, t, ...] = transition.next_achieved_goal
        self.stats.mdp.next_desired_goals[ep, t, ...] = transition.next_desired_goal
        self.stats.mdp.actions[ep, t, ...] = transition.action
        self.stats.mdp.rewards[ep, t, ...] = transition.reward
        self.stats.mdp.dones[ep, t, ...] = transition.done

    def _set_place(self):
        """
        Place dynamic objects (car and target)
        """

        # Place car
        car_pose = self.rng.uniform(self.car_reset_low, self.car_reset_high)
        self.car.set_pose(car_pose)

        # Place target
        target_pose = self.rng.uniform(self.target_reset_low, self.target_reset_high)
        self.target.set_pose(target_pose)

    def _set_place_ghost_car(self, pose: Vector3 | None = None, steers: Vector2 | None = None):
        """
        Place transparent visual shape of the car, at its current pose, to log trajectory.
        """

        # Call with argument to set at desired pose
        if pose is not None:
            pos = [pose[0], pose[1], 0.001]
            orn = p.getQuaternionFromEuler([0, 0, pose[2]])
            pos1, pos2 = steers

        # Get car's pose and steering wheels angles
        else:
            pos, orn = p.getBasePositionAndOrientation(self.car.id, physicsClientId=self.client)
            pos1, _, _, _ = p.getJointState(self.car.id, 4, physicsClientId=self.client)
            pos2, _, _, _ = p.getJointState(self.car.id, 6, physicsClientId=self.client)

        # Load URDF at car's current pose
        idd = p.loadURDF('ghost.urdf', pos, orn, useFixedBase=1, physicsClientId=self.client)

        # Change steering angles for the shape
        p.resetJointState(idd, 4, pos1, physicsClientId=self.client)
        p.resetJointState(idd, 6, pos2, physicsClientId=self.client)

        # Initialize list of ghosts if not initialized
        if self.l_ghost_ids is None:
            self.l_ghost_ids = []

        self.l_ghost_ids.append(idd)
        self.l_ghost_ids.append(idd)

    def _set_reset_joints(self):
        """
        Reset car joints to make velocity and steering null.
        """

        joints = [2, 3, 4, 5, 6, 7]
        for joint in joints:
            p.resetJointState(self.car.id, joint, 0, physicsClientId=self.client)

    def _set_seed(self):
        """
        Set seed to make resets deterministic.
        """

        if self.cfg.env.general.seed is not None:
            self.rng = np.random.default_rng(self.cfg.env.general.seed)
        else:
            self.rng = np.random.default_rng()

    def _set_stats(self):
        """
        Dictionary to hold statistics.
        """

        # Reset stats if current episode number is 1
        if self.episode_num == 0:
            # noinspection PyTypeChecker
            self.stats = EnvStats.new(self.cfg)

    def _set_wait_objects_fall(self):
        """
        Wait for placed objects to fall.
        """

        for _ in range(20):
            p.stepSimulation(physicsClientId=self.client)

    def _set_zero_episode_num(self):
        """
        Zero out (actually set it to '-1') episode counter. Call it after setting the seed.
        """

        self.episode_num = -1

    def _set_zero_timesteps(self):
        """
        Reset MDP timesteps counter.
        """

        self.timestep = 0
        self.bullet_timestep = 0

    def _setup(self) -> None:
        """
        Several initializations and configurations.
        """

        self._setup_seed()
        self._setup_client()
        self._setup_action_space()
        self._setup_observation_space()
        self._setup_action_repeat()
        self._setup_reset_bounds()
        self._setup_reward_weights()
        self._setup_urdf_paths()
        self._setup_ground_plane()
        self._setup_gravity()
        self._setup_car()
        self._setup_target()
        self._setup_obstacles()
        self._setup_save_state()
        self._setup_init_final_pose()
        self._setup_camera()
        self._setup_episode_num()

    def _setup_action_repeat(self):
        """
        Set constant for repeating actions according to frequency.
        """

        self.action_repeat = int(1 / (self.cfg.env.car.f_val * (1 / 240)))

    def _setup_action_space(self) -> None:
        """
        Set up gym action space.
        """

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def _setup_camera(self):
        """
        Set up matrices for image creation.
        """

        # Get camera distance
        if self.cfg.maps.general.map_name is None:
            self.img_crop = self.cfg.env.test.none.img_crop
            self.img_extent = self.cfg.env.test.none.img_extent
            cam_dist = self.cfg.env.test.none.cam_dist
            cam_target = self.cfg.env.test.none.cam_target

        elif self.cfg.maps.general.map_name == 'narrow':
            self.img_crop = self.cfg.env.test.narrow.img_crop
            self.img_extent = self.cfg.env.test.narrow.img_extent
            cam_dist = self.cfg.env.test.narrow.cam_dist
            cam_target = self.cfg.env.test.narrow.cam_target

        else:
            raise NotImplementedError

        # Remove GUI elements and set top view camera config
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0, physicsClientId=self.client)
        p.resetDebugVisualizerCamera(cam_dist, 0, -89.99, cam_target, physicsClientId=self.client)

    def _setup_car(self) -> None:
        """
        Initialize car.
        """

        self.car = Car(self.cfg, self.client)

    def _setup_client(self) -> None:
        """
        Assign a client ID, necessary for vectorized environments.
        """

        self.client = p.connect(p.GUI) if self.cfg.env.general.gui else p.connect(p.DIRECT)

    def _setup_episode_num(self) -> None:
        """
        Initialize episode number with '-1' because calling 'reset' should increment it by one.
        """

        self.episode_num = -1

    def _setup_gravity(self):
        """
        Set the gravity.
        """

        p.setGravity(0, 0, -9.81, physicsClientId=self.client)

    def _setup_ground_plane(self) -> None:
        """
        Load the ground plane.
        """

        p.setAdditionalSearchPath(str(Paths().deps_models))
        self.plane = p.loadURDF("meshes/plane.urdf", useFixedBase=1, physicsClientId=self.client)

    def _setup_ignore_static_warnings(self, info=None):
        """
        Stop linting 'method may be static' or unused parameter by putting this method inside another method.
        """

        pass

    def _setup_init_final_pose(self):
        """
        Set attributes 'init/final_pose' as the initial achieved/desired goal.
        """

        self.init_pose = self.car.get_pose()
        self.final_pose = self.target.get_pose()

    def _setup_observation_space(self) -> None:
        """
        Set up the gym observation space. State is [x y sinOrn cosOrn v steer]. Goal is [x y sinOrn cosOrn].
        """

        self.observation_space = gym.spaces.Dict(
            dict(
                observation=gym.spaces.Box(
                    low=-np.inf, high=+np.inf, shape=(6,)
                ),
                desired_goal=gym.spaces.Box(
                    low=-np.inf, high=+np.inf, shape=(4,)
                ),
                achieved_goal=gym.spaces.Box(
                    low=-np.inf, high=+np.inf, shape=(4,)
                ),
            )
        )

    def _setup_obstacles(self) -> None:
        """
        Load all obstacles.
        """

        # Return if there is no obstacles to set up
        if self.cfg.maps.general.map_name is None:
            return

        # Create map and add to obstacles descriptions lists
        m = Map(self.cfg)

        # Iterate over vertices and orientations creating 'Obstacles'
        self.obstacles = []
        for vertex, orn in zip(m.vertices, m.orientations):
            self.obstacles.append(Obstacle(self.cfg, vertex, orn, self.client))

    def _setup_reset_bounds(self) -> None:
        """
        Set up reset bounds.
        """

        # Aliases
        cp = self.cfg.env.car.pose
        tp = self.cfg.env.target.pose

        # Set car pose
        if isinstance(cp, float | int):
            self.car_reset_low = [-cp, -cp, 0]
            self.car_reset_high = [+cp, +cp, 2 * np.pi]
        elif isinstance(cp, list | tuple | np.ndarray):
            self.car_reset_low = [cp[0], cp[1], cp[2]]
            self.car_reset_high = [cp[0], cp[1], cp[2]]

        # Set target pose
        if isinstance(tp, float | int):
            self.target_reset_low = [-tp, -tp, 0]
            self.target_reset_high = [+tp, +tp, 2 * np.pi]
        elif isinstance(tp, list | tuple | np.ndarray):
            self.target_reset_low = [tp[0], tp[1], tp[2]]
            self.target_reset_high = [tp[0], tp[1], tp[2]]

    def _set_restore_state(self):
        """
        Restore state to maintain contact information.
        """

        p.restoreState(self.save_state, physicsClientId=self.client)

    def _setup_reward_weights(self) -> None:
        """
        Set up the reward weights for each state component.
        """

        self.reward_weights = 0.5 * np.array(
            [
                self.cfg.env.reward.weights[0],
                self.cfg.env.reward.weights[0],
                self.cfg.env.reward.weights[1],
                self.cfg.env.reward.weights[1],
            ]
        )

    def _setup_save_state(self):
        """
        Save all information to guarantee deterministic resetting.
        """

        # Get contact information for car
        self.car.set_pose([0, 0, 0])
        self.target.set_pose([0, 0, 0])
        for _ in range(10):
            p.stepSimulation(physicsClientId=self.client)

        # Save this state
        self.save_state = p.saveState(physicsClientId=self.client)

    def _setup_seed(self):
        """
        Set up random seed for placing car and target. Deterministic only if setting the seed through other method.
        """

        self.rng = np.random.default_rng()

    def _setup_target(self) -> None:
        """
        Initialize target.
        """

        self.target = Target(self.cfg, client=self.client)

    def _setup_urdf_paths(self) -> None:
        """
        Set additional search paths for pybullet.
        """

        p.setAdditionalSearchPath(pd.getDataPath(), physicsClientId=self.client)


class Car:

    def __init__(self, cfg: MasterConfig, client: int = 0):
        """
        Initialize car.
        :param client: pybullet client id
        :param cfg: config object
        """

        self.client = client
        self.cfg = cfg

        self.id = None  # type: None | int

        self._setup()

    def get_achieved_goal(self):
        """
        Get car achieved goal.
        :return: vector [x y sinOrn cosOrn]
        """

        # Get x, y and theta
        pose = self.get_pose()
        state = pose3_to_state4(pose)

        return state

    def get_pose(self):
        """
        Get car position and orientation.
        :return: car pose
        """

        # Get from PyBullet
        pos, orn = p.getLinkState(self.id, 0, physicsClientId=self.client)[:2]

        # Get orientation
        theta = quaternion_to_theta(orn)

        return np.array([pos[0], pos[1], theta])

    def get_state(self):
        """
        Get car state.
        :return: vector [x y sinOrn cosOrn v phi]
        """

        # Get x, y and theta
        pose = self.get_pose()
        state = pose3_to_state4(pose)

        # Get base velocity
        lin_vel = p.getBaseVelocity(self.id, physicsClientId=self.client)
        v = np.linalg.norm([lin_vel[0], lin_vel[1]])

        # Get steering angle
        phi = p.getJointState(self.id, 4, physicsClientId=self.client)[0]

        return np.array([*state, v, phi])

    def get_state_mpc(self):
        """
        Get state for MPC controller (velocity is signed).
        :return: state as [x y theta v phi]
        """

        # Get pose
        pose = self.get_pose()

        # Get velocity with sign
        lin_vel = p.getBaseVelocity(self.id, physicsClientId=self.client)
        vel = np.linalg.norm([lin_vel[0], lin_vel[1]])
        joint_pos, joint_vel, _, _ = p.getJointState(self.id, 2, physicsClientId=self.client)
        sign = 1 if joint_vel > 0 else -1
        vel = sign * vel

        # Get steering angle
        a = self.cfg.env.car.axis_dist
        b = self.cfg.env.car.wheel_dist
        phir, _, _, _ = p.getJointState(self.id, 6)
        phi = np.arctan((2 * a * np.tan(phir)) / (2 * a - b * np.tan(phir)))

        return np.array([*pose, vel, phi])

    def set_pose(self, pose: Vector3):
        """
        Set the target to a certain pose.
        :param pose: vector of 3 components: (x y angle[rad])
        """

        # Use PyBullet API
        x, y = pose[0], pose[1]
        z = self.cfg.env.car.reset_z
        orn_quaternion = p.getQuaternionFromEuler([0, 0, pose[2]])
        p.resetBasePositionAndOrientation(self.id, [x, y, z], orn_quaternion, physicsClientId=self.client)

        # Perform collision detection because we may place the car at different poses to check free regions
        p.performCollisionDetection(physicsClientId=self.client)

    def set_motors_refs_ackermann(self, v_ref: float, phi_ref: float) -> None:
        """
        Move car according to the Ackermann's geometry.
        :param v_ref: desired linear velocity
        :param phi_ref: desired steering angle
        """

        # Use formula to calculate velocities and steering angles
        vrl, vrr, vfl, vfr, phil, phir = get_ackermann_v_rf_lr_phi_lr(v_ref, phi_ref, self.cfg)

        # Set linear velocities
        v_dot_max = self.cfg.env.car.v_dot_max
        p.setJointMotorControl2(self.id, 7, p.VELOCITY_CONTROL, force=v_dot_max, targetVelocity=vfr,
                                physicsClientId=self.client)
        p.setJointMotorControl2(self.id, 5, p.VELOCITY_CONTROL, force=v_dot_max, targetVelocity=vfl,
                                physicsClientId=self.client)
        p.setJointMotorControl2(self.id, 3, p.VELOCITY_CONTROL, force=v_dot_max, targetVelocity=vrr,
                                physicsClientId=self.client)
        p.setJointMotorControl2(self.id, 2, p.VELOCITY_CONTROL, force=v_dot_max, targetVelocity=vfl,
                                physicsClientId=self.client)

        # Set steering angles
        phi_dot_max = self.cfg.env.car.phi_dot_max
        p.setJointMotorControl2(self.id, 6, p.POSITION_CONTROL, maxVelocity=phi_dot_max, targetPosition=phir,
                                physicsClientId=self.client)
        p.setJointMotorControl2(self.id, 4, p.POSITION_CONTROL, maxVelocity=phi_dot_max, targetPosition=phil,
                                physicsClientId=self.client)

    def _setup(self):
        """
        Set up.
        """

        self._setup_id()

    def _setup_id(self):
        """
        Set up urdf.
        """

        p.setAdditionalSearchPath(str(Paths().deps_models), physicsClientId=self.client)
        self.id = p.loadURDF("racecar.urdf", physicsClientId=self.client)

    def _setup_ignore_static_warnings(self):
        """
        Ignore PyCharm static method warnings.
        """

        pass


class Target:

    def __init__(self, cfg: MasterConfig, client: int):
        """
        Initialize target as desired pose.
        """

        self.cfg = cfg
        self.client = client

        self.id = None

        self._setup()

    def get_pose(self):
        """
        Get target pose.
        :return: vector [x y sinTheta cosTheta]
        """

        # Get from PyBullet
        pos, orn = p.getBasePositionAndOrientation(self.id, physicsClientId=self.client)

        # Get orientation
        theta = quaternion_to_theta(orn)

        return np.array([pos[0], pos[1], theta])

    def get_desired_goal(self):
        """
        Get desired goal.
        :return: vector [x y sinTheta cosTheta]
        """

        # Get x, y and theta
        pose = self.get_pose()
        state = pose3_to_state4(pose)

        return state

    def set_pose(self, pose: Vector3):
        """
        Set the target to a certain pose.
        :param pose: vector of 3 components: (x y angle[rad])
        """

        # Use PyBullet API
        x, y = pose[0], pose[1]
        z = self.cfg.env.target.reset_z
        orn_quaternion = p.getQuaternionFromEuler([0, 0, pose[2]])
        p.resetBasePositionAndOrientation(self.id, [x, y, z], orn_quaternion, physicsClientId=self.client)

    def _setup(self):
        """
        Set up.
        """

        self._setup_id()

    def _setup_id(self):
        """
        Set up urdf.
        """

        p.setAdditionalSearchPath(str(Paths().deps_models), physicsClientId=self.client)
        self.id = p.loadURDF('target.urdf', useFixedBase=1, globalScaling=0.025, physicsClientId=self.client)


class Obstacle:

    def __init__(self, cfg: MasterConfig, top_left_bottom_right: Vector4, orn: float, client: int):
        """
        Initialize.
        """

        self.cfg = cfg
        self.top_left_bottom_right = top_left_bottom_right
        self.orn = orn
        self.client = client

        self.id = None

        self._setup()

    def _setup(self):
        """
        Set up.
        """

        self._setup_id()

    def _setup_id(self):
        """
        Set up id.
        """

        # We set up a collision and visual shape
        ln = self.top_left_bottom_right[2] - self.top_left_bottom_right[0]
        w = self.top_left_bottom_right[1] - self.top_left_bottom_right[3]
        h = self.cfg.maps.obstacles.height
        c = self.cfg.maps.obstacles.color
        m = self.cfg.maps.obstacles.mass
        z = self.cfg.maps.obstacles.reset_z

        # Create collision and visual shapes
        collision_shape = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[ln / 2, w / 2, h],
            physicsClientId=self.client,
        )
        visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[ln / 2, w / 2, h],
            rgbaColor=c,
            physicsClientId=self.client,
        )

        # Create a heavy multi-body, so that it can't be moved by the car
        x = self.top_left_bottom_right[0] + ln / 2
        y = self.top_left_bottom_right[3] + w / 2
        orn = p.getQuaternionFromEuler([0, 0, self.orn])
        self.id = p.createMultiBody(
            baseMass=m,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x, y, z],
            baseOrientation=orn,
            physicsClientId=self.client,
        )


class Transition(BaseStats):
    action: Vector2 | None = None
    reward: float | None = None
    done: bool | None = None
    desired_goal: np.ndarray | None = None
    achieved_goal: np.ndarray | None = None
    state: np.ndarray | None = None
    next_desired_goal: np.ndarray | None = None
    next_achieved_goal: np.ndarray | None = None
    next_state: np.ndarray | None = None

    @classmethod
    def new(cls, obs, next_obs, action, reward, done):
        t = Transition()
        t.desired_goal = obs['desired_goal']
        t.achieved_goal = obs['achieved_goal']
        t.state = obs['observation']
        t.next_desired_goal = next_obs['desired_goal']
        t.next_achieved_goal = next_obs['achieved_goal']
        t.next_state = next_obs['observation']
        t.action = action
        t.reward = reward
        t.done = done
        return t
