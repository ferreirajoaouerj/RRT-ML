import torch as t
from numpy.random import Generator
from rich.progress import track
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from rrt_ml.algorithms.base import *
from rrt_ml.utilities.datasets import *
from rrt_ml.utilities.formulas import *
from rrt_ml.utilities.infos import *
from rrt_ml.utilities.maps import *
from rrt_ml.utilities.models import *
from rrt_ml.utilities.stats import *


class SL(Algorithm):
    """
    Algorithm to learn to sample states for motion planning.
    """

    def __init__(self, cfg):
        """
        Initialize.
        """

        super(SL, self).__init__(cfg)
        self.dataset_train = None  # type: None | Dataset
        self.dataset_val = None  # type: None | Dataset
        self.dataloader_train = None  # type: None | DataLoader
        self.dataloader_val = None  # type: None | DataLoader

        self._setup()

    def train(self):
        """
        Train CVAE.
        """

        # Training loop
        for epoch_n in track(range(self.epoch + 1, self.cfg.sl.train.n_epochs), "Training CVAE..."):

            # Initialize list of stats
            t_l, v_l = [], []
            t_kl_l, v_kl_l = [], []
            t_r_l, v_r_l = [], []

            # Train
            for batch_n, sample in enumerate(iter(self.dataloader_train)):
                l, kl, r = self._set_train_on_batch(sample)
                t_l.append(l)
                t_kl_l.append(kl)
                t_r_l.append(r)

            # Validate
            for batch_n, sample in enumerate(iter(self.dataloader_val)):
                l, kl, r = self._set_validate_on_batch(sample)
                v_l.append(l)
                v_kl_l.append(kl)
                v_r_l.append(r)

            # Get all epoch info and set validation loss attributes
            epoch_info = SLEpochInfo.new(epoch_n, t_l, v_l, t_kl_l, v_kl_l, t_r_l, v_r_l)
            self._set_attrs_after_epoch(epoch_info)

            # Save
            self.save_stats(epoch_info)
            self.save_checkpoint()

            # Log
            self.log_console(epoch_info)
            self.log_tensorboard(epoch_info)

        # Set training flag and save attrs
        self.training_over = True
        self._set_save_attributes()

    def test(self):
        """
        Test SL config.
        """

        self._test_plot_losses()
        self._test_plot_multiple_problems()
        self._test_plot_progression()
        pass

    def save_checkpoint(self):
        """
        Save checkpoint.
        """

        # Save attributes
        self._set_save_attributes()

        # Check if validation loss improved
        if self.model_curr_loss < self.model_best_loss:
            # Log
            self.console.print(f"\n[red bold underline]Validation loss decreased from "
                               f"{self.model_best_loss:.2e} to {self.model_curr_loss:.2e}... Saving model...[/]")

            # Save model
            self._set_save_model()

    def save_stats(self, epoch_info: 'SLEpochInfo') -> None:
        """
        Save train statistics.
        :param epoch_info: current epoch info.
        :return: None.
        """

        # Get locals
        ep = epoch_info.epoch_n
        t_l = epoch_info.train_loss
        v_l = epoch_info.val_loss
        t_kl_l = epoch_info.train_kl_loss
        v_kl_l = epoch_info.val_kl_loss
        t_r_l = epoch_info.train_recon_loss
        v_r_l = epoch_info.val_recon_loss
        b = self.cfg.sl.train.batch_size

        # Compute mean loss = mean(epoch_losses) / batch_size
        mean_train_loss = np.array(t_l).mean() / b
        mean_val_loss = np.array(v_l).mean() / b
        mean_train_kl_loss = np.array(t_kl_l).mean() / b
        mean_val_kl_loss = np.array(v_kl_l).mean() / b
        mean_train_recon_loss = np.array(t_r_l).mean() / b
        mean_val_recon_loss = np.array(v_r_l).mean() / b

        # Save to object
        self.stats.train_loss.append(mean_train_loss)
        self.stats.val_loss.append(mean_val_loss)
        self.stats.train_kl_loss.append(mean_train_kl_loss)
        self.stats.val_kl_loss.append(mean_val_kl_loss)
        self.stats.train_recon_loss.append(mean_train_recon_loss)
        self.stats.val_recon_loss.append(mean_val_recon_loss)

        # Generate states and save
        for idx in range(self.cfg.sl.val.n_maps):
            # Get sample
            sample = self.dataset_val[idx]
            y = sample['y']

            # Generate samples
            states = self.get_samples(y, self.cfg.sl.val.n_states, np.random.default_rng())

            # Append
            self.stats.arr_epoch_idx_state_dim[ep, idx, :] = states

        # Save object in file
        self.stats.save_to_file(self.path_stats)

    def log_console(self, epoch_info: 'SLEpochInfo') -> None:
        """
        Log to console
        :param epoch_info: current epoch info.
        :return: None.
        """

        # Get locals
        epoch_n = epoch_info.epoch_n
        t_l = epoch_info.train_loss
        v_l = epoch_info.val_loss
        b = self.cfg.sl.train.batch_size

        # Compute mean loss = mean(epoch_losses) / batch_size
        mean_train_loss = np.array(t_l).mean() / b
        mean_val_loss = np.array(v_l).mean() / b

        self.console.print(
            f"\n[blue bold underline]Epoch:[/blue bold underline] [blue]{epoch_n}[/blue]\t"
            f"[cyan bold underline]Train loss:[/cyan bold underline] [cyan]{mean_train_loss:.2e}[/cyan]\t"
            f"[green bold underline]Val loss:[/green bold underline] [green]{mean_val_loss:.2e}[/green]"
        )

    def log_tensorboard(self, epoch_info: 'SLEpochInfo') -> None:
        """
        Log info to tensorboard.
        :param epoch_info: current epoch info.
        :return: None
        """

        epoch_n = epoch_info.epoch_n
        t_l = epoch_info.train_loss
        v_l = epoch_info.val_loss
        t_kl_l = epoch_info.train_kl_loss
        v_kl_l = epoch_info.val_kl_loss
        t_r_l = epoch_info.train_recon_loss
        v_r_l = epoch_info.val_recon_loss
        b = self.cfg.sl.train.batch_size

        # Compute mean loss = mean(epoch_losses) / batch_size
        mean_train_loss = np.array(t_l).mean() / b
        mean_val_loss = np.array(v_l).mean() / b
        mean_train_kl_loss = np.array(t_kl_l).mean() / b
        mean_val_kl_loss = np.array(v_kl_l).mean() / b
        mean_train_recon_loss = np.array(t_r_l).mean() / b
        mean_val_recon_loss = np.array(v_r_l).mean() / b

        # Write losses to tensorboard
        loss = {'train': mean_train_loss, 'val': mean_val_loss}
        self.tb.add_scalars("loss", loss, epoch_n)
        kl_loss = {'train': mean_train_kl_loss, 'val': mean_val_kl_loss}
        self.tb.add_scalars("kl_loss", kl_loss, epoch_n)
        recon_loss = {'train': mean_train_recon_loss, 'val': mean_val_recon_loss}
        self.tb.add_scalars("recon_loss", recon_loss, epoch_n)

        # Check generation of states
        self.tb.add_figure("generated_poses", self._get_tb_plot_predictions(), epoch_n)

    def get_conditions(self, map_num: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get conditions from validation dataset (sample with index 'map_num').
        :return: obstacles, initial state and final state
        """

        # Maps are grouped with different samples
        # idx = 20 * map_num
        idx = map_num

        # Get whole condition vector
        y = self.dataset_val[idx]['y']

        # Separate
        obstacles = y[0:self.cfg.sl.dim.obstacle]  # type: t.Tensor
        state_i = y[self.cfg.sl.dim.obstacle:(self.cfg.sl.dim.obstacle + self.cfg.sl.dim.state)]  # type: t.Tensor
        state_f = y[-self.cfg.sl.dim.state:]  # type: t.Tensor

        # Convert to numpy array
        obstacles = obstacles.cpu().numpy()  # type: np.ndarray
        state_i = state_i.cpu().numpy()  # type: np.ndarray
        state_f = state_f.cpu().numpy()  # type: np.ndarray

        return obstacles, state_i, state_f

    def get_samples(self, y: np.ndarray | t.Tensor | int, n_samples: int | None, rng: Generator) -> np.ndarray:
        """
        Generate states.
        """

        # Case y is int we get sample from validation set
        if isinstance(y, int):
            y = self.dataset_val[y]['y']

        # First replicate condition vector along the first axis and move to cuda
        if isinstance(y, np.ndarray):
            y = t.tensor(y).repeat((n_samples, 1)).cuda()
        else:
            y = y.repeat((n_samples, 1)).cuda()

        # Turn on eval mode and stop gradient tracking
        self.model.eval()
        with t.no_grad():

            # Sample 'n_samples' from latent vector and concatenate with condition vector
            z = t.tensor(rng.normal(0, 1, (n_samples, self.cfg.sl.dim.latent))).cuda()
            z_and_y = t.cat((z, y), dim=1).float()

            # Decode
            x = self.model.mlp_decoder(z_and_y)

            # Move to cpu and get as numpy array
            x = x.cpu().numpy()

        return x

    def _get_tb_plot_predictions(self):
        """
        Get plot of generated predictions on current epoch
        :return: matplotlib figure to add to tensorboard
        """

        # Get constants (idx is the first validation sample)
        ep = self.epoch
        idx = 0

        # Get states from saved stats
        states = self.stats.arr_epoch_idx_state_dim[ep, idx, :]

        # Initialize map and get plot axis, then get figure from axis
        maps = Map(self.cfg)
        maps.set_add_states(states[:50, :], 'sl')
        fig, ax = maps.get_plot_lines()

        return fig

    def _set_train_on_batch(self, sample: dict) -> Vector3:
        """
        Train model on a batch
        :param sample: batch sample
        :return: total loss, KL loss and reconstruction loss
        """

        # Prepare
        self.model.train()
        self.model_optimizer.zero_grad()

        # Forward
        x, mu, sigma = self.model(sample['x'], sample['y'])

        # Loss and optimize
        loss, kl_loss, recon_loss = self.model.loss(x, sample['x'], mu, sigma)
        loss.backward()
        self.model_optimizer.step()

        return loss.item(), kl_loss.item(), recon_loss.item()

    def _set_validate_on_batch(self, sample: dict) -> Vector3:
        """
        Validate model on a batch.
        :param sample: batch sample.
        :return: total loss, KL loss and reconstruction loss.
        """

        # Prepare model
        self.model.eval()

        # Don't track gradients and get loss
        with t.no_grad():
            x, mu, sigma = self.model(sample['x'], sample['y'])
            loss, kl_loss, recon_loss = self.model.loss(x, sample['x'], mu, sigma)

        return loss.item(), kl_loss.item(), recon_loss.item()

    def _set_attrs_after_epoch(self, epoch_info):
        """
        Set attributes after epoch of training.
        :param epoch_info: current epoch info.
        """

        # Update iteration
        self.epoch = epoch_info.epoch_n

        # Update current loss
        self.model_curr_loss = np.array(epoch_info.val_loss).mean() / self.cfg.sl.train.batch_size

        # Check best loss so far
        if len(self.stats.val_loss) > 1:
            self.model_best_loss = min(self.stats.val_loss)

    def _test_plot_losses(self):
        """
        Plot model training and validation losses.
        """

        # Return if figure exists
        path = self.path_figs / 'sl_erros.png'
        if path.exists():
            return

        # Get figure
        fig, axs = plt.subplots(3, 1, sharex='all', squeeze=True, figsize=(12, 9.6))
        fig.subplots_adjust(hspace=0.25)

        # Plot recon loss
        axs[0].plot(range(1, 50), self.stats.train_recon_loss, label="Treino")
        axs[0].plot(range(1, 50), self.stats.val_recon_loss, label="Validação")
        axs[0].set_title('Erro de Reconstrução')
        axs[0].legend()
        axs[0].grid(True)
        axs[0].set_ylim(None, 0.005)

        # Plot KL loss
        axs[1].plot(range(1, 50), self.stats.train_kl_loss, label="Treino")
        axs[1].plot(range(1, 50), self.stats.val_kl_loss, label="Validação")
        axs[1].set_title('Erro de Divergência KL')
        axs[1].legend()
        axs[1].grid(True)

        # Plot total loss
        axs[2].plot(range(1, 50), self.stats.train_loss, label="Treino")
        axs[2].plot(range(1, 50), self.stats.val_loss, label="Validação")
        axs[2].set_title('Erro Total')
        axs[2].legend()
        axs[2].grid(True)
        axs[2].set_ylim(None, 0.01)
        axs[2].set_xlabel('Época')

        # Save
        fig.savefig(str(path), dpi=600, bbox_inches='tight')

    def _test_plot_progression(self):
        """
        Plot progression of sample generation.
        """

        # Return if figure exists
        path = self.path_figs / 'sl_progresso.png'
        if path.exists():
            return

        # Constants
        map_num = 23

        # Get problem conditions and change config before creating the map
        obstacles, state_i, state_f = self.get_conditions(map_num)
        cfg = self.cfg.copy(deep=True)
        cfg.maps.general.map_name = 'narrow'
        cfg.maps.narrow.narrow1_pos = obstacles[:2]
        cfg.maps.narrow.narrow2_pos = obstacles[2:]
        cfg.env.car.pose = state4_to_pose3(state_i)
        cfg.env.target.pose = state4_to_pose3(state_f)

        # Initialize figure
        fig, axs = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(12, 9.6))  # type: plt.Figure
        fig.subplots_adjust(hspace=-0.2)
        axs = axs.flatten()

        # Which epochs to plot
        epochs = [1, 9, 19, 29, 39, 49]

        # Plot
        for i, epoch in enumerate(epochs):

            # Get states
            states = self.stats.arr_epoch_idx_state_dim[epoch, map_num, :15]

            # Create map and add states
            mapp = Map(cfg)
            mapp.set_add_states(states, 'sl')
            mapp.set_add_states(state_i, 'init')
            mapp.set_add_states(state_f, 'final')
            mapp.get_plot_lines(axs[i])

            # Add title
            axs[i].set_title(f'Época {epoch}')

            # Axes labels
            if i in [0, 3]:
                axs[i].set_ylabel('y [m]')
            if i in [3, 4, 5]:
                axs[i].set_xlabel('x [m]')

        # Save
        fig.savefig(str(path), dpi=600, bbox_inches='tight')

    def _test_plot_multiple_problems(self):
        """
        Get CVAE solution on test problems.
        """

        # Return if figure exists
        path = self.path_figs / 'sl_diversos_problemas.png'
        if path.exists():
            return

        # Initialize figure
        fig, axs = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(12, 9.6))  # type: plt.Figure
        fig.subplots_adjust(hspace=-0.2)
        axs = axs.flatten()

        # Get problems
        problems = [20, 40, 60, 80, 100, 120]

        # Plot
        for i, p in enumerate(problems):

            # Get conditions
            obstacles, state_i, state_f = self.get_conditions(p)
            cfg = self.cfg.copy(deep=True)
            cfg.maps.general.map_name = 'narrow'
            cfg.maps.narrow.narrow1_pos = obstacles[:2]
            cfg.maps.narrow.narrow2_pos = obstacles[2:]
            # cfg.env.car.pose = state4_to_pose3(state_i)
            # cfg.env.target.pose = state4_to_pose3(state_f)
            cfg.env.car.pose = [state_i[0], state_i[1], np.arctan2(state_i[3], state_i[2])]
            cfg.env.target.pose = [state_f[0], state_f[1], np.arctan2(state_f[3], state_f[2])]

            # Get samples
            condition = np.concatenate([obstacles, state_i, state_f])
            states = self.get_samples(condition, 15, np.random.default_rng(0))

            # States are in the form [x y cos sin] but map needs [x y sin cos]
            states = state4_sl_to_state4_rl(states)
            state_i = state4_sl_to_state4_rl(state_i)
            state_f = state4_sl_to_state4_rl(state_f)

            # Create map and add states
            mapp = Map(cfg)
            mapp.set_add_states(states, 'sl')
            mapp.set_add_states(state_i, 'init')
            mapp.set_add_states(state_f, 'final')
            mapp.get_plot_lines(axs[i])

            # Add title
            axs[i].set_title(f'Problema {i+1}')

            # Axes labels
            if i in [0, 3]:
                axs[i].set_ylabel('y [m]')
            if i in [3, 4, 5]:
                axs[i].set_xlabel('x [m]')

        # Save
        fig.savefig(path, dpi=600, bbox_inches='tight')

    def _setup(self):
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

        # Set up SL settings
        self._setup_dataset()
        self._setup_dataloader()
        self._setup_cuda()

    def _setup_checkpoint(self):
        """
        Set up checkpoint.
        """

        # Check if there is a checkpoint
        self.load_checkpoint = self._get_checkpoint_exists_indicator()

        # Load model and attributes if it does
        if self.load_checkpoint:
            self._set_load_attributes()
            self._set_load_model()

    def _setup_cuda(self):
        """
        Set up cuda.
        """

        self.model.to("cuda")

    def _setup_init_model(self):
        """
        Set up CVAE model and optimizer.
        """

        self.model = CVAE(self.cfg)
        self.model_optimizer = Adam(self.model.parameters(), lr=self.cfg.sl.train.lr)

    def _setup_dataloader(self):
        """
        Set up train and validation loaders.
        """

        self.dataloader_train = DataLoader(
            self.dataset_train, self.cfg.sl.train.batch_size, pin_memory=False
        )
        self.dataloader_val = DataLoader(
            self.dataset_val, self.cfg.sl.train.batch_size, pin_memory=False
        )

    def _setup_dataset(self):
        """
        Set up train and validation datasets.
        """

        self.dataset_train = NarrowCVAEDataset(cfg=self.cfg, train=True)
        self.dataset_val = NarrowCVAEDataset(cfg=self.cfg, train=False)

    def _setup_stats(self):
        """
        Set up stats object.
        """

        if self.load_checkpoint:
            self.stats = SLStats.load_from_file(self.path_stats)
        else:
            self.stats = SLStats.new(self.cfg)
            self.stats.save_to_file(self.path_stats)
