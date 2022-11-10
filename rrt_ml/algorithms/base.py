from abc import ABC
from pathlib import Path

import matplotlib
import numpy as np
import torch as t
import yaml
from joblib import load, dump
from rich.console import Console
from torch.nn import Module
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from rrt_ml.utilities.configs import MasterConfig
from rrt_ml.utilities.paths import Paths
from rrt_ml.utilities.stats import BaseStats


class Algorithm(ABC):
    """
    Abstract base class.
    """

    def __init__(self, cfg: MasterConfig):

        # Base
        self.cfg = cfg
        self.console = None  # type: Console | None
        self.epoch = None
        self.load_checkpoint = None  # type: bool | None
        self.model = None  # type: None | Module
        self.model_best_loss = None  # type: None | float
        self.model_optimizer = None  # type: None | Adam
        self.model_curr_loss = None  # type: None | float
        self.path_attrs = None  # type: Path | None
        self.path_figs = None  # type: Path | None
        self.path_model = None  # type: Path | None
        self.path_model_optimizer = None  # type: Path | None
        self.path_stats = None  # type: Path | None
        self.path_tb = None  # type: Path | None
        self.save_attrs = None  # type: list[str, ...] | None
        self.stats = None  # type: BaseStats | None
        self.tb = None  # type: SummaryWriter | None
        self.training_over = None  # type: bool | None

    def train(self, *args, **kwargs):
        """
        Train algorithm / model.
        """
        
        raise NotImplementedError
    
    def test(self, *args, **kwargs):
        """
        Test algorithm / model.
        """
        
        raise NotImplementedError

    def save_checkpoint(self, *args, **kwargs):
        """
        Save algorithm every iteration using this function.
        """

        pass

    def save_stats(self, *args, **kwargs):
        """
        Save training stats using this function.
        """

        pass

    def log_console(self, *args, **kwargs):
        """
        Log iterations on the console using this function.
        """
        
        raise NotImplementedError

    def log_tensorboard(self, *args, **kwargs):
        """
        Log iterations on tensorboard using this function.
        """
        
        raise NotImplementedError

    def _setup(self, *args, **kwargs):
        """
        General _setup.
        """

        # Set paths, load checkpoint, stats, console and tensorboard
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

    def _get_checkpoint_exists_indicator(self):
        """
        Check if checkpoint exists.
        :return: boolean indicating whether checkpoint exists.
        """

        return self.path_attrs.exists()

    def _set_ignore_static_warnings(self):
        """
        Ignore linting errors.
        """

        pass

    def _set_load_attributes(self):
        """
        Load algorithm attributes.
        """

        attrs_dict = load(self.path_attrs)
        for k in attrs_dict.keys():
            if k in self.save_attrs:
                setattr(self, k, attrs_dict[k])

    def _set_load_model(self):
        """
        Load model if it exists.
        """

        # Load model
        self.model.load_state_dict(t.load(self.path_model))

        # Load optimizer and move parameters to cuda
        for state in self.model_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, t.Tensor):
                    state[k] = v.cuda() if t.cuda.is_available() else state[k]

    def _set_save_attributes(self):
        """
        Save attributes to file.
        """

        # Save attributes
        save_dict = {}
        for k, v in self.__dict__.items():
            if k in self.save_attrs:
                save_dict[k] = v
        dump(save_dict, str(self.path_attrs))

    def _set_save_model(self):
        """
        Save model.
        """

        t.save(self.model.state_dict(), self.path_model)
        t.save(self.model_optimizer.state_dict(), self.path_model_optimizer)

    def _setup_attrs_to_save(self, *args, **kwargs):
        """
        Set up list of attributes to save on checkpoint.
        """

        self.save_attrs = ['epoch', 'model_best_loss', 'model_curr_loss', 'training_over']

    def _setup_checkpoint(self, *args, **kwargs):
        """
        Initialize or load checkpoint object.
        """

        pass

    def _setup_console(self):
        """
        Initialize console logger and log initial information.
        """

        # Aliases
        alg = self.cfg.general.algorithm
        name = self.cfg.general.config_name_or_prefix

        # Create object
        self.console = Console(width=150)

        # Ruler
        self.console.rule(), self.console.rule()

        # Log config algorithm and name
        self.console.print(f"[blue bold underline]Algorithm:[/][blue] {alg}[/]", justify='center')
        self.console.print(f"[magenta bold underline]Running config:[/][magenta] {name}[/]", justify='center')

        # Log loading checkpoint
        if self.load_checkpoint:
            self.console.print(f"[red bold underline]Loading checkpoint... [/]", justify='center')

        # Ruler
        self.console.rule(), self.console.rule()

    def _setup_folders(self):
        """
        Set up experiment folders.
        """

        # Get aliases
        alg = self.cfg.general.algorithm
        name = self.cfg.general.config_name_or_prefix

        try:
            Paths().exp(alg, name).mkdir()
        except FileExistsError:
            pass

        try:
            Paths().exp_checkpoint(alg, name).mkdir()
        except FileExistsError:
            pass

        try:
            Paths().exp_stats(alg, name).mkdir()
        except FileExistsError:
            pass

        try:
            Paths().exp_tensorboard(alg, name).mkdir()
        except FileExistsError:
            pass

        try:
            Paths().exp_fig(alg, name).mkdir()
        except FileExistsError:
            pass

    def _setup_init_base_attrs(self):
        """
        Initialize base attributes (epoch, model losses, etc)
        """

        self.epoch = 0
        self.model_curr_loss = np.inf
        self.model_best_loss = np.inf
        self.training_over = False

    def _setup_init_model(self):
        """
        Initialize model.
        """

        pass

    def _setup_paths(self, *args, **kwargs):
        """
        Setup paths for checkpoints, models, etc.
        """

        # Aliases
        alg = self.cfg.general.algorithm
        name = self.cfg.general.config_name_or_prefix

        # Attributes path
        self.path_attrs = Paths().exp_checkpoint(alg, name) / "attrs"

        # Stats path
        self.path_stats = Paths().exp_stats(alg, name) / "stats"

        # Figures path
        self.path_figs = Paths().exp_fig(alg, name)

        # Tensorboard path
        self.path_tb = Paths().exp_tensorboard(alg, name)

        # Model and optimizer paths
        self.path_model = Paths().exp_checkpoint(alg, name) / "model.pt"
        self.path_model_optimizer = Paths().exp_checkpoint(alg, name) / "model_optimizer.pt"

    def _setup_save_config_to_file(self):
        """
        Save config to file.
        """

        # Aliases
        alg = self.cfg.general.algorithm
        name = self.cfg.general.config_name_or_prefix

        # Save config at experiment base folder
        path = Paths().exp(alg, name) / 'config.yaml'
        if not path.exists():
            with open(str(path), 'w') as f:
                yaml.dump(self.cfg.dict(), f)

    def _setup_stats(self, *args, **kwargs):
        """
        Set up stats object.
        """

        pass

    def _setup_tensorboard(self, *args, **kwargs):
        """
        Setup tensorboard logging.
        """

        # Don't show figure window
        # matplotlib.use('Agg')

        if not self.training_over:
            self.tb = SummaryWriter(log_dir=str(self.path_tb))
