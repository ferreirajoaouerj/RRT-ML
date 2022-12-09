from typing import Any
from argparse import Namespace

from pydantic import BaseModel
import yaml

from rrt_ml.runner.run_hyper import run_hyper
from rrt_ml.runner.run_rl import run_rl
from rrt_ml.runner.run_rrt import run_rrt
from rrt_ml.runner.run_sl import run_sl
from rrt_ml.utilities.hints import *
from rrt_ml.utilities.paths import *


class MasterConfig(BaseModel):

    class _env(BaseModel):

        class _car(BaseModel):
            axis_dist: float = 0.325
            f_val: int = 10
            phi_dot_max: float = 20.0
            phi_max: float = 0.7
            pose: float | list[float, float, float] = 1.0
            reset_z: float = 0.001
            wheel_dist: float = 0.2
            wheel_radius: float = 0.05
            v_dot_max: float = 1.0
            v_max: float = 1.0

        class _general(BaseModel):
            gui: bool = False
            max_timestep: int = 100
            seed: int = 0
            stats: bool = False

        class _reward(BaseModel):
            epsilon: float = -0.10
            p_val: float = 0.5
            weights: list[float, float] = [0.75, 0.15]

        class _target(BaseModel):
            pose: float | list[float, float, float] = 1.0
            reset_z: float = 0.0

        class _test(BaseModel):

            class _general(BaseModel):
                ghost_interval: int = 90

            class _narrow(BaseModel):
                cam_dist: float = 8.0
                cam_target: list[float, float, float] = [5, 5, 0]
                img_crop: list[int, int, int, int] = [232, 126, 808, 704]
                img_extent: list[int, int, int, int] = [-1, 11, -1, 11]

            class _none(BaseModel):
                cam_dist: float = 3.35
                cam_target: list[float, float, float] = [0, 0, 0]
                img_crop: list[int, int, int, int] = [176, 71, 864, 759]
                img_extent: list[int, int, int, int] = [-3, 3, -3, 3]

            general: _general = _general()
            narrow: _narrow = _narrow()
            none: _none = _none()

        car: _car = _car()
        general: _general = _general()
        reward: _reward = _reward()
        target: _target = _target()
        test: _test = _test()

    class _general(BaseModel):

        algorithm: str | None = 'rl'
        config_name_or_prefix: str | None = 'default'
        description: str = ""
        is_hyper: bool | None = False
        is_train: bool | None = True

    class _hyperparams(BaseModel):

        class _general(BaseModel):
            name: str = 'h'

        class _rl(BaseModel):

            class _actor(BaseModel):
                l2: list[float, ...] | None = None
                lr: list[float, ...] | None = None
                weight_decay: list[float, ...] | None = None

            class _critic(BaseModel):
                lr: list[float, ...] | None = None

            class _exploration(BaseModel):
                epsilon: list[float, ...] | None = None
                go_epsilon: list[float, ...] | None = None
                go_reset_percent: list[float, ...] | None = None
                initial: list[int, ...] | None = None
                noise: list[float, ...] | None = None
                warm_up: list[int, ...] | None = None

            class _general(BaseModel):
                gamma: list[float, ...] | None = None
                her: list[str, ...] | None = None

            class _net(BaseModel):
                activ: list[str, ...] | None = None
                grad_value_clipping: list[float, ...] | None = None
                layer_norm: list[bool, ...] | None = None
                layers: list[list[int, ...], ...] | None = [[512, 512, 512], [1024, 1024]]

            class _target(BaseModel):
                grad_clip: list[float, ...] | None = None
                update_frac: list[float, ...] | None = None
                update_freq: list[int, ...] | None = None

            class _train(BaseModel):
                batch_size: list[int, ...] | None = [1024, 2048]
                n_timesteps: list[int, ...] | None = None
                optimize_every: list[int, ...] | None = None
                replay_size: list[int, ...] | None = None

            actor: _actor = _actor()
            critic: _critic = _critic()
            exploration: _exploration = _exploration()
            general: _general = _general()
            net: _net = _net()
            target: _target = _target()
            train: _train = _train()

        class _rrt(BaseModel):
            
            class _general(BaseModel):
                add_last_pose: list[bool, ...] | None = None
                condition: list[str | int, ...] | None = None
                controller: list[str, str] | None = ['rl', 'mpc']
                debug: list[bool, ...] | None = None
                delta_t: list[int, ...] | None = None
                distance_fn: list[str, ...] | None = ['rl-ordinal-00', 'rs']
                distance_fn_mode: list[str, ...] | None = None
                epsilon_multiplier: list[float, ...] | None = None
                goal_epsilon: list[float, ...] | None = None
                n_iter: list[int, ...] | None = None
                policy_limitation: list[bool, ...] | None = None
                rewire: list[bool, ...] | None = None
                seed: list[int | None, ...] | None = [i for i in range(9)]

            class _mpc(BaseModel):
                control_cost: list[float, ...] | None = None
                n_horizon: list[int, ...] | None = None
                n_robust: list[int, ...] | None = None
                simulate: list[bool, ...] | None = None
                step_size: list[float, ...] | None = None
                weight_angle: list[float, ...] | None = None
                weight_pos: list[float, ...] | None = None

            class _names(BaseModel):
                rl: list[str, ...] | None = None
                sl: list[str, ...] | None = None

            class _near(BaseModel):
                cutoff_length: list[float, ...] | None = None
                cutoff_timesteps: list[int, ...] | None = None
                max_n_nodes: list[int, ...] | None = None

            class _nearest(BaseModel):
                min_n_timesteps: list[int, ...] | None = None
                min_length: list[float, ...] | None = None

            class _ordinal(BaseModel):  # opt refer to ranges for optuna: [start, stop] or [start, stop, step]
                batch_size: list[int, ...] | None = None
                data_ratio: list[float, ...] | None = None
                layers: list[list[int, ...], ...] | None = None
                max_timesteps: list[int, ...] | None = None
                n_estimators: list[int, ...] | None = None
                opt: list[bool, ...] | None = None
                opt_batch_size: list[list[int, int, int], ...] | None = None
                opt_n_estimators: list[list[int, int, int], ...] | None = None
                opt_n_layers: list[list[int, int], ...] | None = None
                opt_n_units: list[list[int, int, int], ...] | None = None
                use_model_from_experiment: list[str, ...] | None = None

            class _rs(BaseModel):
                curvature: list[float, ...] | None = None
                step_size: list[float, ...] | None = None
                tolerance: list[float, ...] | None = None

            class _sample(BaseModel):
                goal_prob: list[float, ...] | None = None
                learner_prob: list[Any, ...] | None = None

            general: _general = _general()
            mpc: _mpc = _mpc()
            names: _names = _names()
            near: _near = _near()
            nearest: _nearest = _nearest()
            ordinal: _ordinal = _ordinal()
            rs: _rs = _rs()
            sample: _sample = _sample()

        class _sl(BaseModel):

            class _dim(BaseModel):
                latent: list[int, ...] | None = [2, 3]

            class _loss(BaseModel):
                beta: list[float, ...] | None = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]

            class _net(BaseModel):
                mlp_units: list[list[int, ...], ...] | None = [[256, 256], [512, 512]]

            class _train(BaseModel):
                batch_size: list[int, ...] | None = None
                lr: list[float, ...] | None = [0.001, 0.01]
                n_epochs: list[int, ...] | None = None

            dim: _dim = _dim()
            loss: _loss = _loss()
            net: _net = _net()
            train: _train = _train()

        general: _general = _general()
        rl: _rl = _rl()
        rrt: _rrt = _rrt()
        sl: _sl = _sl()

    class _maps(BaseModel):

        class _arena(BaseModel):
            length: float = 10
            offset: list[float, float] = [0, 0]
            width: float = 10

        class _general(BaseModel):
            map_name: None | str = None

        class _narrow(BaseModel):
            narrow1_pos: list[float, float] = [3.5, 4.5]
            narrow2_pos: list[float, float] = [6.5, 4.5]
            passage_width: float = 1.4
            size: float = 10
            wall_width: float = 0.4

        class _obstacles(BaseModel):
            color: list[float, float, float, float] = [0, 0, 0, 1]
            height: float = 0.05
            mass: float = 1000.0
            reset_z: float = 0.05

        arena: _arena = _arena()
        general: _general = _general()
        narrow: _narrow = _narrow()
        obstacles: _obstacles = _obstacles()

    class _rl(BaseModel):

        class _actor(BaseModel):
            l2: float = 0.1
            lr: float = 0.001
            weight_decay: float = 0.0

        class _critic(BaseModel):
            lr: float = 0.001

        class _exploration(BaseModel):
            epsilon: float = 0.1
            go_epsilon: float = 0.1
            go_reset_percent: float = 0.0
            initial: int = 5000
            noise: float = 0.1
            warm_up: int = 2500

        class _general(BaseModel):
            gamma: float = 0.99
            her: str = 'rfaab_1_4_3_1_1'
            just_use_policy: bool = False

        class _net(BaseModel):
            activ: str = 'gelu'
            grad_value_clipping: float = 5.0
            layer_norm: bool = True
            layers: list[int, int, int] = [512, 512, 512]

        class _target(BaseModel):
            grad_clip: float = 5.0
            update_frac: float = 0.05
            update_freq: int = 40

        class _train(BaseModel):
            batch_size: int = 2048
            n_envs: int = 1
            n_timesteps: int = 1000000
            optimize_every: int = 1
            replay_size: int = 1000000

        class _val(BaseModel):
            gui: bool = False
            stats: bool = True
            interval: int = 10000
            n_envs = 1
            n_episodes = 50
            target_pose: float = 2.0

        actor: _actor = _actor()
        critic: _critic = _critic()
        exploration: _exploration = _exploration()
        general: _general = _general()
        net: _net = _net()
        target: _target = _target()
        train: _train = _train()
        val: _val = _val()

    class _rrt(BaseModel):

        class _general(BaseModel):
            add_last_pose: bool = True  # when simulate reaching, use the last node instead of the random node
            condition: int | str = 'random'  # 'random' generate a new planning problem and integer get from val set
            controller: str = 'rl'  # mpc
            debug: bool = False
            delta_t: int = 160
            distance_fn: str = 'rl-ordinal-00'  # rl-ordinal-xx / rl-sim / rs
            distance_fn_mode: str = 'percent'  # or 'difference'
            epsilon_multiplier: float = 1.0
            goal_epsilon: float = -0.5
            n_iter: int = 500
            policy_limitation: bool = False
            rewire: bool = True
            seed: int | None = 0

        class _mpc(BaseModel):
            control_cost: float = 0.01
            n_horizon: int = 100
            n_robust: int = 4
            simulate: bool = False
            step_size: float = 0.01
            weight_angle: float = 0.05
            weight_pos: float = 1.0

        class _names(BaseModel):
            rl: str = 'best'
            sl: str = 'best'

        class _near(BaseModel):
            cutoff_length: float = 1.0
            cutoff_timesteps: int = 240
            max_n_nodes: int = 5000

        class _nearest(BaseModel):
            min_n_timesteps: int = 0
            min_length: float = 0.0

        class _ordinal(BaseModel):  # opt refer to ranges for optuna: [start, stop] or [start, stop, step]
            batch_size: int = 128
            data_ratio: float = 1
            layers: list[int, ...] = [256, 256, 256]
            max_timesteps: int | None = 48
            n_estimators: int = 10
            opt: bool = False
            opt_batch_size: list[int, int, int] = [16, 512, 16]
            opt_n_estimators: list[int, int, int] = [10, 40, 5]
            opt_n_layers: list[int, int] = [1, 3]
            opt_n_units: list[int, int, int] = [16, 512, 16]
            use_model_from_experiment: None | str = 'default'

        class _rs(BaseModel):
            curvature: float = 2.3
            step_size: float = 0.01
            tolerance: float = 0.01

        class _sample(BaseModel):
            goal_prob: float = 0.05
            learner_prob: float | list[float, float, float] = [1.38, -186.44, 161.44]

        general: _general = _general()
        mpc: _mpc = _mpc()
        names: _names = _names()
        near: _near = _near()
        nearest: _nearest = _nearest()
        ordinal: _ordinal = _ordinal()
        rs: _rs = _rs()
        sample: _sample = _sample()

    class _sl(BaseModel):

        class _dim(BaseModel):
            state: int = 4
            obstacle: int = 4
            latent: int = 4

        class _general(BaseModel):
            data_ratio: float = 0.5

        class _loss(BaseModel):
            beta: float = 0.01

        class _net(BaseModel):
            mlp_units: list[int, ...] = [512, 512]

        class _train(BaseModel):
            batch_size: int = 64
            lr: float = 0.001
            n_epochs: int = 50

        class _val(BaseModel):
            n_maps: int = 30
            n_states: int = 1000

        dim: _dim = _dim()
        general: _general = _general()
        loss: _loss = _loss()
        net: _net = _net()
        train: _train = _train()
        val: _val = _val()

    env: _env = _env()
    general: _general = _general()
    hyperparams: _hyperparams = _hyperparams()
    maps: _maps = _maps()
    rl: _rl = _rl()
    rrt: _rrt = _rrt()
    sl: _sl = _sl()

    def deep_copy_change(self, attr: str, val: Any):
        """
        Deep copy config and change attribute.
        :param attr: string representing nested attribute to change, ex: 'env.general.gui'
        :param val: value to set attribute
        :return: new config with changed value
        """

        # Deep copy config to return and make a shallow copy to reach last config level
        cfg = self.copy(deep=True)
        shallow = cfg

        # Go to last config level
        nested_attrs = attr.split('.')
        for att in nested_attrs[:-1]:
            shallow = shallow.__getattribute__(att)

        # Now set the attribute using shallow copy
        shallow.__setattr__(nested_attrs[-1], val)

        return cfg

    def run(self, algorithm_to_run='rl', train_or_test='train', hyperparam_search_or_test=False):
        """
        Run algorithm following this config.
        :param algorithm_to_run: run 'rl', 'sl' or 'rrt'.
        :param train_or_test: train module or test it.
        :param hyperparam_search_or_test: if True and train then perform hyper search. If test, compare after search.
        """

        self.general.algorithm = algorithm_to_run
        self.general.is_train = True if train_or_test == 'train' else False
        self.general.is_hyper = hyperparam_search_or_test

        if not self.general.is_hyper:
            match self.general.algorithm:
                case 'rl':
                    run_rl(self)
                case 'rrt':
                    run_rrt(self)
                case 'sl':
                    run_sl(self)
                case _:
                    raise NotImplementedError

        else:
            run_hyper(self)

    def save(self):
        """
        Save config to yaml file.
        """

        # Save config at experiment base folder
        path = Paths().configs / self.general.config_name_or_prefix
        with open(str(path), 'w') as f:
            yaml.dump(self.dict(), f)

    @classmethod
    def load(cls, args):
        """
        Load config from file.
        :return: new config file with overridden attributes
        """

        # Check if we need to load default
        is_default = args.config is None

        # Change name for default config
        if is_default:
            config_name_or_prefix = 'h' if args.hyper else 'default'
        else:
            config_name_or_prefix = args.config

        # Get parameters from cmd line args and path to config
        is_train = args.train
        is_hyper = args.hyper
        match True:
            case args.rl:
                algorithm = 'rl'
            case args.sl:
                algorithm = 'sl'
            case args.rrt:
                algorithm = 'rrt'
            case _:
                raise NotImplementedError

        # Load dict from config to override default config dict
        if not is_default:

            # If config file exists, load it
            path = Paths().configs / 'config_name_or_prefix'
            if path.exists():
                with open(str(path)) as f:
                    overrider_dict = yaml.safe_load(f)

            # Maybe it doesn't exist but hyperparameter config do
            else:

                # Check if experiment folder exists
                path = Paths().exp(algorithm, config_name_or_prefix)
                if path.exists():

                    # Experiment folder exists, load config from yaml
                    with open(str(path / 'config.yaml'), 'r') as f:
                        overrider_dict = yaml.safe_load(f)

        else:
            overrider_dict = {}
        merged_dict = cls().dict() | overrider_dict

        # Change general attributes of merged config
        merged_cfg = cls(**merged_dict)
        merged_cfg.general.algorithm = algorithm
        merged_cfg.general.config_name_or_prefix = config_name_or_prefix
        merged_cfg.general.is_train = is_train
        merged_cfg.general.is_hyper = is_hyper

        # Post process merged config
        merged_cfg = cls()._set_defaults(merged_cfg)

        # Override again
        merged_dict = merged_cfg.dict() | overrider_dict

        # Override default config with custom config and return it
        return cls()._set_defaults(merged_cfg)

    @classmethod
    def load_from_experiment(cls, alg, name):
        """
        Load config directly from experiment folder.
        :param alg: algorithm name
        :param name: name of the experiment
        """

        # Get path to experiment config
        path = Paths().exp(alg, name) / 'config.yaml'

        # Check existence
        assert path.exists(), "Trying to load config from experiment that doest not exists."
        
        # Get dict
        with open(str(path)) as f:
            overrider_dict = yaml.safe_load(f)
        
        # Get merged dict
        merged_dict = cls().dict() | overrider_dict

        # Change general attributes of merged config
        return cls(**merged_dict)

    @classmethod
    def _set_defaults(cls, config: 'MasterConfig'):
        """
        Change defaults according to algorithm
        :param config: master config.
        :return: config with default parameters changed.
        """

        match config.general.algorithm:

            # Set defaults for SL
            case 'sl':

                # Map should not be 'None' as in RL
                config.maps.general.map_name = 'narrow'

            # Set defaults for RRT
            case 'rrt':

                # Map should not be 'None' as in RL
                config.maps.general.map_name = 'narrow'

                # Pose is fixed (no random resetting)
                config.env.car.pose = [1, 5, 0]
                config.env.target.pose = [9, 5, 0]

                # Must track stats
                config.env.general.stats = True

                # Action repeat is implemented in RRT don't need to change
                config.env.car.f_val = 240
                config.env.general.max_timestep = 10000

                # Don't need envs from RL agent
                config.rl.general.just_use_policy = True

                # For testing we need GUI to take screenshot
                if not config.general.is_train:
                    config.env.general.gui = True

            case _:
                pass

        return config

