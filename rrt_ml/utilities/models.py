import numpy as np
import pandas as pd
import torch as t
from coral_pytorch.dataset import corn_label_from_logits
from torch.nn import functional as f

from rrt_ml.utilities.configs import *
from rrt_ml.utilities.paths import *


class CVAE(t.nn.Module):

    def __init__(self, cfg: MasterConfig):
        """
        Initialize.
        :param cfg: SL config.
        """

        super(CVAE, self).__init__()

        # Attributes
        self.cfg = cfg

        self.device = None
        self.mlp_encoder_shared = None
        self.mlp_encoder_mu = None
        self.mlp_encoder_sigma = None
        self.mlp_decoder = None
        self.beta_norm = None

        self._setup()

    def forward(self, x, y):
        """
        Forward.
        :param x: state vector.
        :param y: conditions vector.
        :return: x_hat, mu and log_sigma.
        """

        # Concatenate input and condition passing through encoder MLP
        x_and_y = t.cat((x, y), dim=1)
        encoder_shared_out = self.mlp_encoder_shared(x_and_y)

        # Get mu and log sigma outputs
        mu = self.mlp_encoder_mu(encoder_shared_out)
        log_sigma = self.mlp_encoder_sigma(encoder_shared_out)

        # Transform log_sigma to sigma
        sigma = t.exp(log_sigma / 2)

        # Re-parametrization trick
        epsilon = t.randn_like(mu)
        z = mu + epsilon * sigma

        # Concatenate latent vector and condition
        z_and_y = t.cat((z, y), dim=1)

        # Pass through decoder MLP
        x_hat = self.mlp_decoder(z_and_y)

        return x_hat, mu, log_sigma

    def loss(self, x_recon, x_real, mu, log_sigma):
        """
        Calculate loss.
        """

        # Individual losses
        kl_loss = self.beta_norm * t.mean(
            t.sum(t.exp(log_sigma) + mu ** 2 - 1 - log_sigma, dim=1), dim=0
        )
        recon_loss = f.mse_loss(x_recon, x_real)

        # Weighted loss
        loss = kl_loss + recon_loss

        return loss, kl_loss, recon_loss

    def _setup(self):
        """
        Setup model.
        """

        self._setup_device()
        self._setup_encoder()
        self._setup_decoder()
        self._setup_beta()
        self._setup_float()

    def _setup_device(self):
        """
        Setup device, GPU if cuda available.
        """

        # Check if cuda available
        self.device = "cuda" if t.cuda.is_available() else "cpu"

    def _setup_encoder(self):
        """
        Setup encoder MLP.
        """

        # Encoder MLP
        mlp_encoder = []
        for i, n_units in enumerate(self.cfg.sl.net.mlp_units):
            mlp_encoder.extend([t.nn.LazyLinear(n_units), t.nn.ReLU()])
        self.mlp_encoder_shared = t.nn.Sequential(*mlp_encoder)

        # Encoder mu and sigma
        self.mlp_encoder_mu = t.nn.LazyLinear(self.cfg.sl.dim.latent)
        self.mlp_encoder_sigma = t.nn.LazyLinear(self.cfg.sl.dim.latent)

    def _setup_decoder(self):
        """
        Setup decoder MLP.
        """

        # Decoder
        mlp_decoder = []
        for i, n_units in enumerate(self.cfg.sl.net.mlp_units):
            mlp_decoder.extend([t.nn.LazyLinear(n_units), t.nn.ReLU()])
        mlp_decoder.append(t.nn.LazyLinear(self.cfg.sl.dim.state))
        self.mlp_decoder = t.nn.Sequential(*mlp_decoder)

    def _setup_beta(self):
        """
        Setup KL balancing.
        """

        self.beta_norm = self.cfg.sl.loss.beta * self.cfg.sl.dim.latent / self.cfg.sl.dim.state

    def _setup_float(self):
        """
        Change model parameters to 32 bit.
        """

        self.float()


class Ordinal(t.nn.Module):

    def __init__(self, cfg: MasterConfig = MasterConfig()):
        """
        Initialize ordinal MLP regressor.
        """

        super().__init__()

        self.cfg = cfg
        self.timesteps_range = None  # type: None | list[int, int]

        self._setup()

    def forward(self, x):
        """
        Forward.
        :param x: input.
        :return: output.
        """

        return self.net(x)

    def get_distance(self, x: np.ndarray):
        """
        Predict distance.
        :return: distance predictions.
        """

        # Convert to tensor
        x_t = t.tensor(x, dtype=t.float32).to("cuda")

        # Forward
        with t.no_grad():
            self.eval()
            y = self(x_t)
            y = corn_label_from_logits(y)

        # Add minimum timesteps from dataset to get actual values
        pred = y.cpu().numpy().flatten() - self.timesteps_range[0]

        return pred

    def _setup(self):
        """
        Set up.
        """

        self._setup_timesteps_range()
        self._setup_net()
        self._setup_cuda()

    def _setup_cuda(self):
        """
        Set model to cuda.
        """

        self.cuda()

    def _setup_timesteps_range(self):
        """
        Get min max timesteps from data.
        """

        # Set column names
        col_names = ['x', 'y', 'sin_theta', 'cos_theta', 'success', 'time', 'timesteps', 'length']

        # Read data
        path = Paths().data_rl_distance / self.cfg.rrt.names.rl / 'train.csv'
        df = pd.read_csv(path, names=col_names, index_col=None)

        # Get min and max values for timesteps
        self.timesteps_range = [df['timesteps'].min(), df['timesteps'].max()]

    def _setup_net(self):
        """
        Set up net architecture.
        """

        # Get number of classes
        max_timesteps = int(self.timesteps_range[1] - self.timesteps_range[0] + 1)

        # Net
        units = []
        for i, n_units in enumerate(self.cfg.rrt.ordinal.layers):
            units.extend([t.nn.LazyLinear(n_units), t.nn.GELU()])
        units.append(t.nn.LazyLinear(max_timesteps))
        self.net = t.nn.Sequential(*units)

        # Convert to float
        self.float()
        self.net.float()


class OrdinalEnsemble:

    def __init__(self, cfg: MasterConfig = MasterConfig()):
        """
        Initialize.
        :param cfg: master config.
        """

        self.models = []

    def get_distance(self, x):
        """
        Get distance prediction from ensemble.
        :param x: input array [[x_car y_car sin_car cos_car]]
        :return: distance prediction
        """

        # Convert to tensor
        x_t = t.tensor(x, dtype=t.float32, device='cuda', requires_grad=False)

        # Initialize array
        batch_size = x_t.size()[0]
        all_pred = np.zeros(shape=(20, batch_size))

        # Forward
        with t.no_grad():
            for i, model in enumerate(self.models):
                model.eval()
                all_pred[i, :] = corn_label_from_logits(model(x_t)).detach().cpu().numpy()

        # Get mean value
        all_pred = np.mean(all_pred, axis=0)

        # Add minimum timesteps from dataset to get actual values
        all_pred = all_pred - self.models[0].timesteps_range[0]

        return all_pred
