from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import interp1d

from rrt_ml.algorithms.rl import *
from rrt_ml.utilities.configs import *


def run_rl(cfg: MasterConfig):

    # Train
    if cfg.general.is_train:

        # Initialize RL
        rl = RL(cfg)
        rl.train()

    # Test
    else:
        rl = RL(cfg)
        rl.test()

    # Test
    # elif cfg.test:

        # AgentUtils.plot_trajectory_top_view(target_pose=[-1, 0, np.pi])
        # AgentUtils.plot_success_rate()

        # Initialize RL
        # rl = RL(cfg)

        # Test episodes
        # rl.test_multiple_episodes()

        # Generate data
        # rl.get_distance_data()
        # rl.get_distance_data()

        # Test distance model
        # rl.test_distance_model()
        # rl.test_distance_model_corn()

        # Test trajectory
        # rl.test_plot_trajectory_top_view_variables(1)
        # rl.test_get_top_view_image()

        # Hyperparameter search graph
        # plot_hyperparameter_search(cfg)

        # Plot reward and entropy
        # plot_reward_entropy(cfg)


def plot_hyperparameter_search(cfg):
    """
    Results from hyperparameter search.
    :param cfg: experiment configuration.
    """

    # Experiments path
    base_path = Path(__file__).resolve().parents[2]
    exp_path = base_path / "experiments" / "rl"
    file_path = base_path.parents[0] / "latex" / "figs" / "apdx" / "rl_hiper.png"

    # Figures
    fig, axs = plt.subplots(nrows=4, ncols=2, sharex="all", sharey="all")
    fig.set_size_inches(16, 22)
    axs = axs.flatten()

    # Loop
    ax_counter = 0
    for param1 in [1]:
        for param2 in [1, 2]:
            for param3 in [1, 2]:

                # Name
                prefix = "a" + str(param1) + str(param2) + str(param3)

                # Load test rewards
                path = exp_path / prefix / (prefix + "__Test__Episode_rewards.csv")
                df = pd.read_csv(path)  # type: pd.DataFrame
                time = df["wall_time"].to_numpy()
                step = df["step"].to_numpy()
                rewards = df["Test/Episode_rewards"].to_numpy()

                # Fix arrays for interpolation
                rewards = rewards[:-1]
                step = step[:-1]

                # Load entropy
                path = exp_path / prefix / (prefix + "__Explore__ag_kde_entropy.csv")
                df = pd.read_csv(path)

                # Smoothing
                rewards = _smooth(rewards, 0.8)

                # Plot rewards
                axs[ax_counter].plot(step, rewards)
                axs[ax_counter].tick_params(axis="y", labelcolor="b")
                axs[ax_counter].yaxis.set_major_locator(
                    MaxNLocator(nbins=5, integer=True)
                )

                # Plot entropy
                ax2 = axs[ax_counter].twinx()
                x = df["step"].to_numpy()
                y = df["Explore/ag_kde_entropy"].to_numpy()
                f = interp1d(x, y)
                ax2.plot(step, f(step), color="g")
                ax2.tick_params(axis="y", labelcolor="g")
                ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

                # Set title
                axs[ax_counter].title.set_text(
                    f"Hiperparâmetro {param1}{param2}{param3}"
                )

                # Increment counter
                ax_counter += 1

    # Save
    fig.savefig(file_path, dpi=600)


def _smooth(series, weight):
    last = series[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in series:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed
