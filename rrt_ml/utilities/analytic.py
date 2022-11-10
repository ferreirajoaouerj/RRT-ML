import math

from casadi import cos, sin, tan
from do_mpc.controller import MPC
from do_mpc.model import Model
from do_mpc.simulator import Simulator
from scipy.spatial.distance import euclidean

from rrt_ml.deps.libraries.python_robotics_reeds_shepp import *
from rrt_ml.utilities.configs import *
from rrt_ml.utilities.formulas import *
from rrt_ml.utilities.hints import *
from rrt_ml.utilities.maps import *


# ORIGINAL (DO NOT CHANGE)
class MPCRS:
    """
    Model predictive controller with Redd-Shepp paths.
    """

    def __init__(self, cfg: MasterConfig):
        """
        Initialize.
        """

        self.cfg = cfg

        self.controller = None  # type: MPC | None
        self.desired_goal = None  # type: np.ndarray | None
        self.rs_xs = None  # type: list[float, ...] | None
        self.rs_ys = None  # type: list[float, ...] | None
        self.rs_yaws = None  # type: list[float, ...] | None
        self.rs_idx = None  # type: int | None
        self.tvp_template = None

        self._setup()

    def _setup(self):
        """
        Set up MPC.
        """

        # Set arbitrary value so that 'get_action' calls '_set_new_target' first time
        self.desired_goal = np.array([1000.0, 1000.0, 1000.0, 1000.0])

    def get_action(self, obs: dict[str, Vector4 | Vector6]):
        """
        Get env action with MPC controller.
        :param obs: current env observation for goal conditioned tasks
        :return: action vector
        """

        # If desired goal changed, we need to set a new target for the MPC
        if np.all(self.desired_goal != obs['desired_goal']):
            self._set_new_target(obs)
            self.controller.x0 = state4_to_pose3(obs['achieved_goal']).reshape(-1, 1)
            self.simulator.x0 = state4_to_pose3(obs['achieved_goal']).reshape(-1, 1)
            self.controller.set_initial_guess()

        # Get current car and target pose
        car_pose = state6_to_pose3(obs['observation'])
        target_pose = [self.rs_xs[self.rs_idx], self.rs_ys[self.rs_idx], self.rs_yaws[self.rs_idx]]

        # Check distance between car and target positions
        distance = euclidean(car_pose[0:2], target_pose[0:2])

        # If distance is small increase index, changing next target pose (tvp function depends on index) for the MPC
        if distance < self.cfg.rrt.rs.tolerance:

            # Check to go over the path length
            if self.rs_idx < (len(self.rs_xs) - 1):
                self.rs_idx += 1

        # Make state column vector
        state = car_pose.reshape(-1, 1)

        # Get control
        u = self.controller.make_step(state).flatten()
        self.simulator.make_step(u.reshape(-1, 1))

        # Now we need to convert to [-1, 1] range
        v_ref = scale_to_range(u[0], [-self.cfg.env.car.v_max, +self.cfg.env.car.v_max], [-1, 1])
        phi_ref = scale_to_range(u[1], [-self.cfg.env.car.phi_max, +self.cfg.env.car.phi_max], [-1, 1])

        return np.array([v_ref, phi_ref])

    def get_distance(self, node_from: Node, node_to: Node):
        """
        Calculate distance between nodes using RS path's length.
        :param node_from: start node
        :param node_to: final node
        :return: distance
        """

        # Get poses
        pose_init = node_from.pose
        pose_final = node_to.pose

        # Get RS path
        curvature = self.cfg.rrt.rs.curvature
        step_size = self.cfg.rrt.rs.step_size
        xs, ys, yaws, modes, lengths = reeds_shepp_path_planning(*pose_init, *pose_final, curvature, step_size)

        # Handle cases where could not calculate route
        if xs is None:
            lengths = [np.inf]

        return np.sum(np.abs(lengths))

    def _set_new_target(self, obs: dict):
        """
        Function to be called when setting a new desired goal: sets RS paths and function to update MPC cost.
        :param obs: gym env observation
        """

        # Update desired goal so that this set up is not called again
        self.desired_goal = obs['desired_goal']

        # Get current car and target pose to calculate waypoints
        car_pose = state6_to_pose3(obs['observation'])
        target_pose = state4_to_pose3(obs['desired_goal'])

        # Get list of waypoints
        curvature = self.cfg.rrt.rs.curvature
        step_size = self.cfg.rrt.rs.step_size
        xs, ys, yaws, modes, lengths = reeds_shepp_path_planning(*car_pose, *target_pose, curvature, step_size)
        self.rs_xs = xs[1:]
        self.rs_ys = ys[1:]
        self.rs_yaws = yaws[1:]

        # Initialize index
        self.rs_idx = 0

        # Get constants from config
        axis_dist = self.cfg.env.car.axis_dist
        n_horizon = self.cfg.rrt.mpc.n_horizon
        n_robust = self.cfg.rrt.mpc.n_robust
        step_size = self.cfg.rrt.mpc.step_size
        v_max = self.cfg.env.car.v_max
        v_min = -v_max
        phi_max = self.cfg.env.car.phi_max
        phi_min = -phi_max
        control_cost = self.cfg.rrt.mpc.control_cost

        # Initialize dynamics model
        model_type = 'continuous'  # either 'discrete' or 'continuous'
        model = Model(model_type)

        # Set state variables
        model.set_variable('_x', 'x')
        model.set_variable('_x', 'y')
        theta = model.set_variable('_x', 'theta')

        # Set control variables
        v = model.set_variable(var_type='_u', var_name='v')
        phi = model.set_variable(var_type='_u', var_name='phi')

        # Time varying parameters
        model.set_variable(var_type='_tvp', var_name='rs_x')
        model.set_variable(var_type='_tvp', var_name='rs_y')
        model.set_variable(var_type='_tvp', var_name='rs_yaw')

        # Set equations
        model.set_rhs('x', v * cos(theta))
        model.set_rhs('y', v * sin(theta))
        model.set_rhs('theta', v * tan(phi) / axis_dist)

        # Setup
        model.setup()

        # Initialize optimizer
        mpc = MPC(model)
        mpc.set_param(n_horizon=n_horizon, t_step=step_size, n_robust=n_robust, store_full_solution=True,
                      nlpsol_opts={'ipopt.print_level': 0, 'ipopt.sb': 'yes','print_time': 0})

        # Set state variables bound
        mpc.bounds['lower', '_x', 'theta'] = -2 * np.pi
        mpc.bounds['upper', '_x', 'theta'] = +2 * np.pi

        # Set control variables bound
        mpc.bounds['lower', '_u', 'v'] = v_min
        mpc.bounds['upper', '_u', 'v'] = v_max
        mpc.bounds['lower', '_u', 'phi'] = phi_min
        mpc.bounds['upper', '_u', 'phi'] = phi_max

        # Set cost for control input (avoid oscillations)
        mpc.set_rterm(v=control_cost, phi=control_cost)

        # Get template and define function
        self.tvp_template = mpc.get_tvp_template()

        def tvp_fun(t_now):
            if len(self.rs_xs[self.rs_idx:]) <= n_horizon:
                for k in range(n_horizon + 1):
                    self.tvp_template['_tvp', k, 'rs_x'] = self.rs_xs[-1]
                    self.tvp_template['_tvp', k, 'rs_y'] = self.rs_ys[-1]
                    self.tvp_template['_tvp', k, 'rs_yaw'] = self.rs_yaws[-1]
            else:
                for k in range(n_horizon + 1):
                    self.tvp_template['_tvp', k, 'rs_x'] = self.rs_xs[self.rs_idx]
                    self.tvp_template['_tvp', k, 'rs_y'] = self.rs_ys[self.rs_idx]
                    self.tvp_template['_tvp', k, 'rs_yaw'] = self.rs_yaws[self.rs_idx]
            return self.tvp_template

        # Set tvp fun
        mpc.set_tvp_fun(tvp_fun)

        # Finally, we need to add the m and l terms to cost function, but we do it when settings the target pose
        self.controller = mpc

        # Retrieve symbolic variables from model
        x = self.controller.model._x['x']
        y = self.controller.model._x['y']
        theta = self.controller.model._x['theta']
        x_target = self.controller.model.tvp['rs_x']
        y_target = self.controller.model.tvp['rs_y']
        theta_target = self.controller.model.tvp['rs_yaw']

        # Set cost for 'm' and 'l' term
        wp = self.cfg.rrt.mpc.weight_pos
        wa = self.cfg.rrt.mpc.weight_angle
        mterm = wp * (x - x_target) ** 2 + wp * (y - y_target) ** 2 + wa * (theta - theta_target) ** 2
        lterm = wp * (x - x_target) ** 2 + wp * (y - y_target) ** 2 + wa * (theta - theta_target) ** 2
        self.controller.set_objective(mterm=mterm, lterm=lterm)

        # Call setup to finalize
        self.controller.setup()

        # Set simulator
        self.simulator = Simulator(model)
        self.simulator.set_param(t_step=step_size)

        # TVP
        tvp_sim = self.simulator.get_tvp_template()

        def tvp_sim_fun(t_now):
            tvp_sim['rs_x'] = self.rs_xs[self.rs_idx]
            tvp_sim['rs_y'] = self.rs_ys[self.rs_idx]
            tvp_sim['rs_yaw'] = self.rs_yaws[self.rs_idx]
            return tvp_sim

        self.simulator.set_tvp_fun(tvp_sim_fun)
        self.simulator.setup()

    def _set_rs_path(self, obs: dict):
        """
        Set new RS path.
        :param obs: env observation
        """

        # Update desired goal so that this set up is not called again
        self.desired_goal = obs['desired_goal']

        # Get current car and target pose to calculate waypoints
        car_pose = state6_to_pose3(obs['observation'])
        target_pose = state4_to_pose3(obs['desired_goal'])

        # Get list of waypoints
        curvature = self.cfg.rrt.rs.curvature
        step_size = self.cfg.rrt.rs.step_size
        xs, ys, yaws, modes, lengths = reeds_shepp_path_planning(*car_pose, *target_pose, curvature, step_size)
        self.rs_xs = xs[1:]
        self.rs_ys = ys[1:]
        self.rs_yaws = yaws[1:]

        # Initialize index
        self.rs_idx = 0

# TEST
# class MPCRS:
#     """
#     Model predictive controller with Redd-Shepp paths.
#     """
#
#     def __init__(self, cfg: MasterConfig):
#         """
#         Initialize.
#         """
#
#         self.cfg = cfg
#
#         self.controller = None  # type: MPC | None
#         self.desired_goal = None  # type: np.ndarray | None
#         self.rs_xs = None  # type: list[float, ...] | None
#         self.rs_ys = None  # type: list[float, ...] | None
#         self.rs_yaws = None  # type: list[float, ...] | None
#         self.rs_idx = None  # type: int | None
#         self.tvp_template = None
#
#         self._setup()
#
#     def _setup(self):
#         """
#         Set up MPC.
#         """
#
#         # Set arbitrary value so that 'get_action' calls '_set_new_target' first time
#         self.desired_goal = np.array([1000.0, 1000.0, 1000.0, 1000.0])
#
#     def get_action(self, obs: dict[str, Vector4 | Vector6]):
#         """
#         Get env action with MPC controller.
#         :param obs: current env observation for goal conditioned tasks
#         :return: action vector
#         """
#
#         # If desired goal changed, we need to set a new target for the MPC
#         if np.all(self.desired_goal != obs['desired_goal']):
#             self._set_new_target(obs)
#             self.controller.x0 = obs['mpc_state'].reshape(-1, 1)
#             self.simulator.x0 = obs['mpc_state'].reshape(-1, 1)
#             self.controller.set_initial_guess()
#
#         # Get current car and target pose
#         car_pose = state6_to_pose3(obs['observation'])
#         target_pose = [self.rs_xs[self.rs_idx], self.rs_ys[self.rs_idx], self.rs_yaws[self.rs_idx]]
#
#         # Check distance between car and target positions
#         distance = euclidean(car_pose[0:2], target_pose[0:2])
#
#         # If distance is small increase index, changing next target pose (tvp function depends on index) for the MPC
#         if distance < self.cfg.rrt.rs.tolerance:
#
#             # Check to go over the path length
#             if self.rs_idx < (len(self.rs_xs) - 1):
#                 self.rs_idx += 1
#
#         # Make state column vector
#         state = obs['mpc_state'].reshape(-1, 1)
#
#         # Get control
#         u = self.controller.make_step(state).flatten()
#         self.simulator.make_step(u.reshape(-1, 1))
#
#         # Now we need to convert to [-1, 1] range
#         v_ref = scale_to_range(u[0], [-self.cfg.env.car.v_max, +self.cfg.env.car.v_max], [-1, 1])
#         phi_ref = scale_to_range(u[1], [-self.cfg.env.car.phi_max, +self.cfg.env.car.phi_max], [-1, 1])
#
#         return np.array([v_ref, phi_ref])
#
#     def get_distance(self, node_from: Node, node_to: Node):
#         """
#         Calculate distance between nodes using RS path's length.
#         :param node_from: start node
#         :param node_to: final node
#         :return: distance
#         """
#
#         # Get poses
#         pose_init = node_from.pose
#         pose_final = node_to.pose
#
#         # Get RS path
#         curvature = self.cfg.rrt.rs.curvature
#         step_size = self.cfg.rrt.rs.step_size
#         xs, ys, yaws, modes, lengths = reeds_shepp_path_planning(*pose_init, *pose_final, curvature, step_size)
#
#         # Handle cases where could not calculate route
#         if xs is None:
#             lengths = [np.inf]
#
#         return np.sum(np.abs(lengths))
#
#     def _set_new_target(self, obs: dict):
#         """
#         Function to be called when setting a new desired goal: sets RS paths and function to update MPC cost.
#         :param obs: gym env observation
#         """
#
#         # Update desired goal so that this set up is not called again
#         self.desired_goal = obs['desired_goal']
#
#         # Get current car and target pose to calculate waypoints
#         car_pose = state6_to_pose3(obs['observation'])
#         target_pose = state4_to_pose3(obs['desired_goal'])
#
#         # Get list of waypoints
#         curvature = self.cfg.rrt.rs.curvature
#         step_size = self.cfg.rrt.rs.step_size
#         xs, ys, yaws, modes, lengths = reeds_shepp_path_planning(*car_pose, *target_pose, curvature, step_size)
#         self.rs_xs = xs[1:]
#         self.rs_ys = ys[1:]
#         self.rs_yaws = yaws[1:]
#
#         # Initialize index
#         self.rs_idx = 0
#
#         # Get constants from config
#         axis_dist = self.cfg.env.car.axis_dist
#         n_horizon = self.cfg.rrt.mpc.n_horizon
#         n_robust = self.cfg.rrt.mpc.n_robust
#         step_size = self.cfg.rrt.mpc.step_size
#         v_max = self.cfg.env.car.v_max
#         v_min = -v_max
#         phi_max = self.cfg.env.car.phi_max
#         phi_min = -phi_max
#         control_cost = self.cfg.rrt.mpc.control_cost
#
#         # Initialize dynamics model
#         model_type = 'continuous'  # either 'discrete' or 'continuous'
#         model = Model(model_type)
#
#         # Set state variables
#         model.set_variable('_x', 'x')
#         model.set_variable('_x', 'y')
#         theta = model.set_variable('_x', 'theta')
#         v = model.set_variable('_x', 'v')
#         phi = model.set_variable('_x', 'phi')
#
#         # Set control variables
#         u_v = model.set_variable(var_type='_u', var_name='u_v')
#         u_phi = model.set_variable(var_type='_u', var_name='u_phi')
#
#         # Time varying parameters
#         model.set_variable(var_type='_tvp', var_name='rs_x')
#         model.set_variable(var_type='_tvp', var_name='rs_y')
#         model.set_variable(var_type='_tvp', var_name='rs_yaw')
#
#         # Set equations
#         tau_v = 0.05
#         k_v = 1.02
#         tau_phi = 0.05
#         k_phi = 1.0
#         model.set_rhs('x', v * cos(theta))
#         model.set_rhs('y', v * sin(theta))
#         model.set_rhs('theta', v * tan(phi) / axis_dist)
#         model.set_rhs('v', v*-(1/tau_v) + u_v*(k_v/tau_v))
#         model.set_rhs('phi', phi*-(1/tau_phi) + u_phi*(k_phi/tau_phi))
#
#         # Setup
#         model.setup()
#
#         # Initialize optimizer
#         mpc = MPC(model)
#         mpc.set_param(n_horizon=n_horizon, t_step=step_size, n_robust=n_robust, store_full_solution=True,
#                       nlpsol_opts={'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0})
#
#         # Set state variables bound
#         mpc.bounds['lower', '_x', 'theta'] = -2 * np.pi
#         mpc.bounds['upper', '_x', 'theta'] = +2 * np.pi
#
#         # Set control variables bound
#         mpc.bounds['lower', '_u', 'u_v'] = v_min
#         mpc.bounds['upper', '_u', 'u_v'] = v_max
#         mpc.bounds['lower', '_u', 'u_phi'] = phi_min
#         mpc.bounds['upper', '_u', 'u_phi'] = phi_max
#
#         # Set cost for control input (avoid oscillations)
#         mpc.set_rterm(u_v=control_cost, u_phi=control_cost)
#
#         # Get template and define function
#         self.tvp_template = mpc.get_tvp_template()
#
#         def tvp_fun(t_now):
#             if len(self.rs_xs[self.rs_idx:]) <= n_horizon:
#                 for k in range(n_horizon + 1):
#                     self.tvp_template['_tvp', k, 'rs_x'] = self.rs_xs[-1]
#                     self.tvp_template['_tvp', k, 'rs_y'] = self.rs_ys[-1]
#                     self.tvp_template['_tvp', k, 'rs_yaw'] = self.rs_yaws[-1]
#             else:
#                 for k in range(n_horizon + 1):
#                     self.tvp_template['_tvp', k, 'rs_x'] = self.rs_xs[self.rs_idx + k]
#                     self.tvp_template['_tvp', k, 'rs_y'] = self.rs_ys[self.rs_idx + k]
#                     self.tvp_template['_tvp', k, 'rs_yaw'] = self.rs_yaws[self.rs_idx + k]
#             return self.tvp_template
#
#         # Set tvp fun
#         mpc.set_tvp_fun(tvp_fun)
#
#         # Finally, we need to add the m and l terms to cost function, but we do it when settings the target pose
#         self.controller = mpc
#
#         # Retrieve symbolic variables from model
#         x = self.controller.model._x['x']
#         y = self.controller.model._x['y']
#         theta = self.controller.model._x['theta']
#         x_target = self.controller.model.tvp['rs_x']
#         y_target = self.controller.model.tvp['rs_y']
#         theta_target = self.controller.model.tvp['rs_yaw']
#
#         # Set cost for 'm' and 'l' term
#         wp = self.cfg.rrt.mpc.weight_pos
#         wa = self.cfg.rrt.mpc.weight_angle
#         mterm = wp * (x - x_target) ** 2 + wp * (y - y_target) ** 2 + wa * (theta - theta_target) ** 2
#         lterm = wp * (x - x_target) ** 2 + wp * (y - y_target) ** 2 + wa * (theta - theta_target) ** 2
#         self.controller.set_objective(mterm=mterm, lterm=lterm)
#
#         # Call setup to finalize
#         self.controller.setup()
#
#         # Set simulator
#         self.simulator = Simulator(model)
#         self.simulator.set_param(t_step=step_size)
#
#         # TVP
#         tvp_sim = self.simulator.get_tvp_template()
#
#         def tvp_sim_fun(t_now):
#             tvp_sim['rs_x'] = self.rs_xs[self.rs_idx]
#             tvp_sim['rs_y'] = self.rs_ys[self.rs_idx]
#             tvp_sim['rs_yaw'] = self.rs_yaws[self.rs_idx]
#             return tvp_sim
#
#         self.simulator.set_tvp_fun(tvp_sim_fun)
#         self.simulator.setup()
#
#     def _set_rs_path(self, obs: dict):
#         """
#         Set new RS path.
#         :param obs: env observation
#         """
#
#         # Update desired goal so that this set up is not called again
#         self.desired_goal = obs['desired_goal']
#
#         # Get current car and target pose to calculate waypoints
#         car_pose = state6_to_pose3(obs['observation'])
#         target_pose = state4_to_pose3(obs['desired_goal'])
#
#         # Get list of waypoints
#         curvature = self.cfg.rrt.rs.curvature
#         step_size = self.cfg.rrt.rs.step_size
#         xs, ys, yaws, modes, lengths = reeds_shepp_path_planning(*car_pose, *target_pose, curvature, step_size)
#         self.rs_xs = xs[1:]
#         self.rs_ys = ys[1:]
#         self.rs_yaws = yaws[1:]
#
#         # Initialize index
#         self.rs_idx = 0
