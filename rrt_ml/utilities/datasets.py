import numpy as np
import pandas as pd
import torch as t
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from rrt_ml.utilities.configs import *
from rrt_ml.utilities.paths import *


class NarrowCVAEDataset(Dataset):

    def __init__(self, cfg: MasterConfig, train: bool):
        """
        Initialize.
        """

        self.cfg = cfg
        self.train = train
        self.data_ratio = cfg.sl.general.data_ratio
        self.path = None
        self.x = None
        self.y = None

        self._setup()

    def __getitem__(self, index: int) -> dict:
        """
        Get sample.
        :param index: sample index.
        :return: sample (x and y fields).
        """

        sample = dict()
        sample['x'] = t.tensor(self.x[index, :], dtype=t.float32, device='cuda')
        sample['y'] = t.tensor(self.y[index, :], dtype=t.float32, device='cuda')

        return sample

    def __len__(self):
        """
        Len of dataset.
        :return: len of dataset.
        """

        return self.x.shape[0]

    def _setup(self):
        """
        Set up dataset.
        """

        self._setup_path()
        self._setup_arrays()

    def _setup_path(self):
        """
        Setup appropriate csv path.
        """

        if self.train:
            self.path = Paths().data_sl_narrow_train_csv
        else:
            self.path = Paths().data_sl_narrow_val_csv

    def _setup_arrays(self):
        """
        Set up data into two arrays x (states) and y (conditions).
        """

        # Read from master file
        arr = pd.read_csv(self.path, sep=',').to_numpy()

        # Constants
        y_dim = 2 * self.cfg.sl.dim.state + self.cfg.sl.dim.obstacle
        x_dim = self.cfg.sl.dim.state

        # Split into x and y taking the ratio (assume data start at index 1 and is y then x)
        stop_at = int(self.cfg.sl.general.data_ratio * len(arr))
        self.x = arr[:stop_at, -x_dim:]
        self.y = arr[:stop_at, 1:(y_dim + 1)]


class DistanceOrdinalDataset(Dataset):

    def __init__(self,
                 cfg: MasterConfig,
                 split_type: str,
                 scaler: None | StandardScaler = None,
                 timesteps_range: None | list[int, int] | str = None):
        """
        Initialize.
        """

        self.cfg = cfg
        self.split_type = split_type
        self.scaler = scaler
        self.timesteps_range = timesteps_range

        self.x = None
        self.y = None

        self._setup()

    def __getitem__(self, index):
        """
        Get sample.
        :param index: sample index.
        :return: sample as namedtuple (x and y fields).
        """

        x = t.tensor(self.x[index, :], dtype=t.float32, device="cuda")
        y = t.tensor(self.y[index], dtype=t.float32, device="cuda")

        return [x, y]

    def __len__(self):
        """
        Len of dataset.
        :return:
        """

        return self.x.shape[0]

    def _setup(self):
        """
        Set up.
        """

        self._setup_path()
        self._setup_arrays()
        self._setup_scaling()

    def _setup_arrays(self):
        """
        Set up data into two arrays x (states) and y (conditions).
        """

        col_names = ['x', 'y', 'sin_theta', 'cos_theta', 'success', 'time', 'timesteps', 'length']

        # Read and remove some columns
        df = pd.read_csv(self.path, names=col_names, index_col=None)
        df.drop(columns=['success', 'time', 'length'], axis=1, inplace=True)

        # Remove above cutoff
        df = df[df['timesteps'] < self.cfg.rrt.ordinal.max_timesteps]

        # Get or set min and max timestep values
        if self.timesteps_range is None:
            self.timesteps_range = [df['timesteps'].min(), df['timesteps'].max()]

        # Scale target to range
        df['timesteps'] = df['timesteps'] - self.timesteps_range[0]

        # Convert to numpy array and get only part of the data        
        stop_at = int(self.cfg.rrt.ordinal.data_ratio * len(df))
        self.x = df.iloc[:stop_at, :-1].to_numpy()
        self.y = df.iloc[:stop_at, -1].to_numpy().astype(int)

    def _setup_path(self):
        """
        Set up appropriate csv path.
        """

        match self.split_type:
            case 'train':
                self.path = Paths().data_rl_distance / self.cfg.rrt.names.rl / "train.csv"
            case 'val':
                self.path = Paths().data_rl_distance / self.cfg.rrt.names.rl / "val.csv"
            case 'test':
                self.path = Paths().data_rl_distance / self.cfg.rrt.names.rl / "test.csv"
            case _:
                raise NotImplementedError

    def _setup_scaling(self):
        """
        Set up data scaling.
        """

        # Instantiate and fit scaler for training
        if self.scaler is None:
            self.scaler = StandardScaler().fit(self.x)

        # Transform x data
        self.x = self.scaler.transform(self.x)

