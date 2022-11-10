from coral_pytorch.losses import corn_loss

from rrt_ml.utilities.configs import *
from rrt_ml.utilities.hints import *


def get_ackermann_v_rf_lr_phi_lr(v_ref: float, phi_ref: float, cfg: MasterConfig) -> Vector6:
    """
    Ackermann formula to get velocity for each wheel and steering angles.
    :param v_ref: desired linear velocity
    :param phi_ref: desired steering angle
    :param cfg: master config that contains car parameters
    :return: [vrl vrr vfl vfr phil phir] (f: forward, r: rear/right, l: left)
    """

    # Clip maximum steering angle
    if abs(phi_ref) > cfg.env.car.phi_max:
        phi_ref = phi_ref / (abs(phi_ref)) * cfg.env.car.phi_max

    # V to rad/s
    v_ref = np.clip(v_ref, -cfg.env.car.v_max, +cfg.env.car.v_max)
    v_ref = v_ref / cfg.env.car.wheel_radius
    if phi_ref == 0:
        r = 99999999
    else:
        r = cfg.env.car.axis_dist / (np.tan(phi_ref))

    # Right and left wheel
    phir = np.arctan(cfg.env.car.axis_dist / (r + cfg.env.car.wheel_dist / 2))
    phil = np.arctan(cfg.env.car.axis_dist / (r - cfg.env.car.wheel_dist / 2))

    # Target velocity for wheels
    vrr = v_ref * (1 + cfg.env.car.wheel_dist / (2 * r))
    vrl = v_ref * (1 - cfg.env.car.wheel_dist / (2 * r))
    vfr = vrr / (np.cos(phir))
    vfl = vrl / (np.cos(phil))

    return vrl, vrr, vfl, vfr, phil, phir


def quaternion_to_theta(quaternion: Vector4) -> float:
    """
    Convert quaternion to theta (orientation in the xy-plane)
    :param quaternion: orientation from bullet
    :return: theta angle
    """

    w = quaternion[3]
    theta = 2 * np.arccos(w)
    theta = theta * quaternion[2] / np.abs(quaternion[2]) if quaternion[2] != 0 else theta

    return theta


def path_length(xs, ys):
    """
    Get path length using arc length formula.
    :param xs: list of xs
    :param ys: list of ys
    :return: path length
    """

    d = 0
    for i, _ in enumerate(zip(xs[:-1], ys[:-1])):
        d += np.sqrt((xs[i]-xs[i+1])**2 + (ys[i]-ys[i+1])**2)

    return d


def pose3_to_state4(pose: Vector3) -> Vector4:
    """
    Unpack angle into sine and cosine, maintaining x and y coordinates.
    :param pose: [x y theta]
    :return: [x y sin_theta cos_theta]
    """

    return np.array([pose[0], pose[1], np.sin(pose[2]), np.cos(pose[2])])


def pose3_to_state4_sl(pose: Vector3 | np.ndarray) -> Vector4 | np.ndarray:
    """
    State in SL is in the form [x y cos sin]
    :param pose: pose as [x y theta]
    :return: state in SL form
    """

    # ndarray  case

    if isinstance(pose, np.ndarray):

        if len(pose.shape) == 1:
            cos = np.cos(pose[2])
            sin = np.sin(pose[2])
            state_sl = np.array([pose[0], pose[1], cos, sin])

        elif len(pose.shape) == 2:
            cos = np.cos(pose[:, 2])
            sin = np.sin(pose[:, 2])
            state_sl = np.zeros(shape=(pose.shape[0], 4))
            state_sl[:, 0] = pose[:, 0]
            state_sl[:, 1] = pose[:, 1]
            state_sl[:, 2] = cos
            state_sl[:, 3] = sin

        else:
            raise NotImplementedError
    else:
        state_sl = pose3_to_state4_sl(np.array(pose))

    return state_sl


def scale_to_range(x: float, src_range: Vector, dst_range: Vector):
    """
    Scale a number from a source range to a destination range.
    :param x: input number
    :param src_range: source range
    :param dst_range: destination range
    :return: scaled vector
    """

    return (x - src_range[0]) * (dst_range[1] - dst_range[0]) / (src_range[1] - src_range[0]) + dst_range[0]


def state4_to_pose3(state: Vector4) -> Vector3:
    """
    Convert [x y sin_theta cos_theta] to [x y theta].
    :param state: vector with unpacked angle
    :return: pose
    """

    return np.array([state[0], state[1], np.arctan2(state[2], state[3])])


def state6_to_pose3(state: Vector6) -> Vector3:
    """
    Convert [x y sin_theta cos_theta v phi] to [x y theta]
    :param state: vector with 6 components, including velocity and steering angle
    :return: corresponding position and orientation
    """

    return np.array([state[0], state[1], np.arctan2(state[2], state[3])])


def state4_sl_to_state4_rl(state_sl: Vector4 | np.ndarray) -> Vector4 | np.ndarray:
    """
    State in SL is in the form [x y cos sin], and in RL is [x y sin cos]
    :param state_sl: state on SL dataset
    :return: state in RL form
    """

    # Single state case
    if len(state_sl.shape) == 1:
        cos = state_sl[2]
        sin = state_sl[3]
        state_rl = np.array([state_sl[0], state_sl[1], sin, cos])

    elif len(state_sl.shape) == 2:
        cos = state_sl[:, 2]
        sin = state_sl[:, 3]
        state_rl = np.copy(state_sl)
        state_rl[:, 2] = sin
        state_rl[:, 3] = cos

    else:
        raise NotImplementedError

    return state_rl


def state4_sl_to_pose3(state_sl: Vector4 | np.ndarray) -> Vector3 | np.ndarray:
    """
    State in SL is in the form [x y cos sin], so we convert to [x y theta]
    :param state_sl: state on SL dataset
    :return: pose [x y theta]
    """

    # Single state case
    if len(state_sl.shape) == 1:
        cos = state_sl[2]
        sin = state_sl[3]
        state_rl = np.array([state_sl[0], state_sl[1], np.arctan2(sin, cos)])

    elif len(state_sl.shape) == 2:
        cos = state_sl[:, 2]
        sin = state_sl[:, 3]
        state_rl = np.zeros(shape=(state_sl.shape[0], 3))
        state_rl[:, 0] = state_sl[:, 0]
        state_rl[:, 1] = state_sl[:, 1]
        state_rl[:, 2] = np.arctan2(sin, cos)

    else:
        raise NotImplementedError

    return state_rl


def transform_to_origin(obs=None, node_from=None, node_to=None):
    """
    Transform target pose to relative pose wrt car.
    :param obs: env observation
    :param node_from: node of car origin
    :param node_to: node of target pose
    :return: new target pose
    """

    # Handle args and get poses
    if obs is None:
        assert node_from is not None and node_to is not None
        car_pose = node_from.pose
        target_pose = node_to.pose
    else:
        assert node_from is None and node_to is None
        car_pose = state4_to_pose3(obs['achieved_goal'])
        target_pose = state4_to_pose3(obs['desired_goal'])
    
    # Constants
    x = target_pose[0]
    y = target_pose[1]
    ct = np.cos(-car_pose[2])
    st = np.sin(-car_pose[2])

    # Translate target
    x = x - car_pose[0]
    y = y - car_pose[1]

    # 2D rotation
    x_new = x*ct - y*st
    y_new = y*ct + x*st
    theta_new = target_pose[2] - car_pose[2]

    return np.array([x_new, y_new, theta_new])


