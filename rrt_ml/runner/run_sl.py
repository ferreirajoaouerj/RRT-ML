import pickle
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MaxNLocator

from rrt_ml.algorithms.sl import *
from rrt_ml.utilities.configs import *


def run_sl(cfg: MasterConfig):

    sample_learner = SL(cfg)

    # Train
    if cfg.general.is_train:
        sample_learner.train()

    # Test
    else:
        sample_learner.test()


def compare_hypers(cfg, name1, name2):
    """
    Compare two configs.
    :param name1: config 1
    :param name2: config 2
    """

    # Matplotlib params
    mpl.rcParams["font.size"] = 10

    # Experiments path
    base_path = Path(__file__).resolve().parents[2]
    exp_path = base_path / "experiments" / "sl"
    file_path = (
        base_path.parents[0] / "latex" / "figs" / "results" / "hiper_compare.png"
    )

    # Initialize figure
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex="all", sharey="all")
    ax = ax.flatten()
    plt.tight_layout()

    # Axes counter
    ax_counter1 = 0
    ax_counter2 = 0
    ax_counter3 = 0

    path1 = exp_path / name1
    path2 = exp_path / name2
    with open(path1 / "stats.pickle", "rb") as file:
        stats1 = pickle.load(file)
    with open(path2 / "stats.pickle", "rb") as file:
        stats2 = pickle.load(file)

    # Plot
    for i, stats in enumerate([stats1, stats2]):

        # Aliases
        x1, y1, x2, y2 = stats["list_cond_obstacles_epoch"][49][0, :]
        ww = cfg.wall_width
        pw = cfg.passage_width

        # Left rectangle
        lower_left = Polygon(
            [
                (x1 - ww / 2, 0),
                (x1 + ww / 2, 0),
                (x1 + ww / 2, y1 - pw / 2),
                (x1 - ww / 2, y1 - pw / 2),
            ]
        )
        upper_left = Polygon(
            [
                (x1 - ww / 2, y1 + pw / 2),
                (x1 + ww / 2, y1 + pw / 2),
                (x1 + ww / 2, 10),
                (x1 - ww / 2, 10),
            ]
        )

        # Right rectangle
        lower_right = Polygon(
            [
                (x2 - ww / 2, 0),
                (x2 + ww / 2, 0),
                (x2 + ww / 2, y2 - pw / 2),
                (x2 - ww / 2, y2 - pw / 2),
            ]
        )
        upper_right = Polygon(
            [
                (x2 - ww / 2, y2 + pw / 2),
                (x2 + ww / 2, y2 + pw / 2),
                (x2 + ww / 2, 10),
                (x2 - ww / 2, 10),
            ]
        )
        patches = [lower_left, lower_right, upper_left, upper_right]

        ax[i].add_collection(
            PatchCollection(patches, edgecolors="k", facecolors="k")
        )  # type: plt.Axes

        # Add limits
        ax[ax_counter1].set_xlim(0, 10)
        ax[ax_counter1].set_ylim(0, 10)

        # Adjust aspect ratio
        ax[i].set_aspect("equal", adjustable="box")
        # ax[i].axis('equal')

        # Add initial and final states
        x1, y1, dx1, dy1, x2, y2, dx2, dy2 = stats["list_cond_states_epoch"][49][0, :]
        ax[ax_counter1].arrow(x1, y1, 0.3 * dx1, 0.3 * dy1, width=0.1, ec="g", fc="g")
        ax[ax_counter1].arrow(x2, y2, 0.3 * dx2, 0.3 * dy2, width=0.1, ec="r", fc="r")

        # Add generated states
        for state in stats["list_gen_states_epoch"][49][0:30]:
            norm = np.sqrt(state[2] ** 2 + state[3] ** 2)
            ax[ax_counter1].arrow(
                state[0],
                state[1],
                0.2 * state[2] / norm,
                0.2 * state[3] / norm,
                width=0.01,
                head_width=0.1,
                head_length=0.1,
                fc="b",
                ec="b",
            )

        # Set title
        name = name1 if i == 0 else name2
        ax[i].title.set_text(f"Hiperparâmetro {name}")

    # Save figure
    fig.savefig(file_path, dpi=600)


def hyperparameter_search_samples(cfg):
    """
    Graph for the various hyperparameters.
    :param cfg: experiment config.
    """

    # Matplotlib params
    mpl.rcParams["font.size"] = 14

    # Experiments path
    base_path = Path(__file__).resolve().parents[2]
    exp_path = base_path / "experiments" / "sl"
    file_path1 = base_path.parents[0] / "latex" / "figs" / "apdx" / "sl_hiper1.png"
    file_path2 = base_path.parents[0] / "latex" / "figs" / "apdx" / "sl_hiper2.png"
    file_path3 = base_path.parents[0] / "latex" / "figs" / "apdx" / "sl_hiper3.png"

    # Initialize figure 1
    fig1, axs1 = plt.subplots(nrows=4, ncols=2, sharex="all", sharey="all")
    fig1.set_size_inches(16, 22)
    axs1 = axs1.flatten()
    plt.tight_layout()

    # Initialize figure 2
    fig2, axs2 = plt.subplots(nrows=4, ncols=2, sharex="all", sharey="all")
    fig2.set_size_inches(16, 22)
    axs2 = axs2.flatten()
    plt.tight_layout()

    # Initialize figure 3
    fig3, axs3 = plt.subplots(nrows=4, ncols=2, sharex="all", sharey="all")
    fig3.set_size_inches(16, 22)
    axs3 = axs3.flatten()
    plt.tight_layout()

    # Axes counter
    ax_counter1 = 0
    ax_counter2 = 0
    ax_counter3 = 0

    # Go to experiment folder
    for param1 in [1, 2]:
        for param2 in [1, 2]:
            for param3 in [1, 2]:
                for param4 in [1, 2, 3]:
                    path = exp_path / (
                        "a" + str(param1) + str(param2) + str(param3) + str(param4)
                    )

                    # Load stats file
                    with open(path / "stats.pickle", "rb") as file:
                        stats = pickle.load(file)

                    # Aliases
                    x1, y1, x2, y2 = stats["list_cond_obstacles_epoch"][49][0, :]
                    ww = cfg.wall_width
                    pw = cfg.passage_width

                    # Left rectangle
                    lower_left = Polygon(
                        [
                            (x1 - ww / 2, 0),
                            (x1 + ww / 2, 0),
                            (x1 + ww / 2, y1 - pw / 2),
                            (x1 - ww / 2, y1 - pw / 2),
                        ]
                    )
                    upper_left = Polygon(
                        [
                            (x1 - ww / 2, y1 + pw / 2),
                            (x1 + ww / 2, y1 + pw / 2),
                            (x1 + ww / 2, 10),
                            (x1 - ww / 2, 10),
                        ]
                    )

                    # Right rectangle
                    lower_right = Polygon(
                        [
                            (x2 - ww / 2, 0),
                            (x2 + ww / 2, 0),
                            (x2 + ww / 2, y2 - pw / 2),
                            (x2 - ww / 2, y2 - pw / 2),
                        ]
                    )
                    upper_right = Polygon(
                        [
                            (x2 - ww / 2, y2 + pw / 2),
                            (x2 + ww / 2, y2 + pw / 2),
                            (x2 + ww / 2, 10),
                            (x2 - ww / 2, 10),
                        ]
                    )
                    patches = [lower_left, lower_right, upper_left, upper_right]

                    # Figure 1
                    if ax_counter1 < 8:

                        # Add patches
                        axs1[ax_counter1].add_collection(
                            PatchCollection(patches)
                        )  # type: plt.Axes

                        # Add limits
                        axs1[ax_counter1].set_xlim(0, 10)
                        axs1[ax_counter1].set_ylim(0, 10)

                        # Adjust aspect ratio
                        # axs1[ax_counter1].set_aspect('equal', adjustable='box')
                        # axs1[ax_counter1].axis('equal')

                        # Add initial and final states
                        x1, y1, dx1, dy1, x2, y2, dx2, dy2 = stats[
                            "list_cond_states_epoch"
                        ][49][0, :]
                        axs1[ax_counter1].arrow(
                            x1, y1, 0.3 * dx1, 0.3 * dy1, width=0.1, fc="g"
                        )
                        axs1[ax_counter1].arrow(
                            x2, y2, 0.3 * dx2, 0.3 * dy2, width=0.1, fc="r"
                        )

                        # Add generated states
                        for state in stats["list_gen_states_epoch"][49][0:30]:
                            norm = np.sqrt(state[2] ** 2 + state[3] ** 2)
                            axs1[ax_counter1].arrow(
                                state[0],
                                state[1],
                                0.2 * state[2] / norm,
                                0.2 * state[3] / norm,
                                width=0.01,
                                head_width=0.1,
                                head_length=0.1,
                                fc="k",
                            )

                        # Set title
                        axs1[ax_counter1].title.set_text(
                            f"Hiperparâmetro {param1}{param2}{param3}{param4}"
                        )

                        # Next subplot
                        ax_counter1 += 1

                    # Figure 2
                    else:

                        if ax_counter2 < 8:

                            # Add patches
                            axs2[ax_counter2].add_collection(
                                PatchCollection(patches)
                            )  # type: plt.Axes

                            # Add limits
                            axs2[ax_counter2].set_xlim(0, 10)
                            axs2[ax_counter2].set_ylim(0, 10)

                            # Adjust aspect ratio
                            # axs2[ax_counter2].set_aspect('equal')
                            # axs2[ax_counter2].axis('equal')

                            # Add initial and final states
                            x1, y1, dx1, dy1, x2, y2, dx2, dy2 = stats[
                                "list_cond_states_epoch"
                            ][49][0, :]
                            axs2[ax_counter2].arrow(
                                x1, y1, 0.3 * dx1, 0.3 * dy1, width=0.1, fc="g"
                            )
                            axs2[ax_counter2].arrow(
                                x2, y2, 0.3 * dx2, 0.3 * dy2, width=0.1, fc="r"
                            )

                            # Add generated states
                            for state in stats["list_gen_states_epoch"][49][0:30]:
                                norm = np.sqrt(state[2] ** 2 + state[3] ** 2)
                                axs2[ax_counter2].arrow(
                                    state[0],
                                    state[1],
                                    0.2 * state[2] / norm,
                                    0.2 * state[3] / norm,
                                    width=0.01,
                                    head_width=0.1,
                                    head_length=0.1,
                                    fc="k",
                                )

                            # Set title
                            axs2[ax_counter2].title.set_text(
                                f"Hiperparâmetro {param1}{param2}{param3}{param4}"
                            )

                            # Next subplot
                            ax_counter2 += 1

                        else:

                            # Add patches
                            axs3[ax_counter3].add_collection(
                                PatchCollection(patches)
                            )  # type: plt.Axes

                            # Add limits
                            axs3[ax_counter3].set_xlim(0, 10)
                            axs3[ax_counter3].set_ylim(0, 10)

                            # Adjust aspect ratio
                            # axs2[ax_counter2].set_aspect('equal')
                            # axs2[ax_counter2].axis('equal')

                            # Add initial and final states
                            x1, y1, dx1, dy1, x2, y2, dx2, dy2 = stats[
                                "list_cond_states_epoch"
                            ][49][0, :]
                            axs3[ax_counter3].arrow(
                                x1, y1, 0.3 * dx1, 0.3 * dy1, width=0.1, fc="g"
                            )
                            axs3[ax_counter3].arrow(
                                x2, y2, 0.3 * dx2, 0.3 * dy2, width=0.1, fc="r"
                            )

                            # Add generated states
                            for state in stats["list_gen_states_epoch"][49][0:30]:
                                norm = np.sqrt(state[2] ** 2 + state[3] ** 2)
                                axs3[ax_counter3].arrow(
                                    state[0],
                                    state[1],
                                    0.2 * state[2] / norm,
                                    0.2 * state[3] / norm,
                                    width=0.01,
                                    head_width=0.1,
                                    head_length=0.1,
                                    fc="k",
                                )

                            # Set title
                            axs3[ax_counter3].title.set_text(
                                f"Hiperparâmetro {param1}{param2}{param3}{param4}"
                            )

                            # Next subplot
                            ax_counter3 += 1

    # Save figure
    fig1.savefig(file_path1, dpi=600)
    fig2.savefig(file_path2, dpi=600)
    fig3.savefig(file_path3, dpi=600)


def hyperparameter_search_loss(cfg):
    """
    experiment config.
    :param cfg: experiment config.
    """

    # Matplotlib params
    mpl.rcParams["font.size"] = 14

    # Experiments path
    base_path = Path(__file__).resolve().parents[2]
    exp_path = base_path / "experiments" / "sl"
    file_path1 = base_path.parents[0] / "latex" / "figs" / "apdx" / "sl_hiper_loss1.png"
    file_path2 = base_path.parents[0] / "latex" / "figs" / "apdx" / "sl_hiper_loss2.png"
    file_path3 = base_path.parents[0] / "latex" / "figs" / "apdx" / "sl_hiper_loss3.png"

    # Initialize figure 1
    fig1, axs1 = plt.subplots(nrows=4, ncols=2, sharex="all")
    fig1.set_size_inches(16, 22)
    axs1 = axs1.flatten()
    # plt.tight_layout()

    # Initialize figure 2
    fig2, axs2 = plt.subplots(nrows=4, ncols=2, sharex="all")
    fig2.set_size_inches(16, 22)
    axs2 = axs2.flatten()
    # plt.tight_layout()

    # Initialize figure 3
    fig3, axs3 = plt.subplots(nrows=4, ncols=2, sharex="all")
    fig3.set_size_inches(16, 22)
    axs3 = axs3.flatten()
    # plt.tight_layout()

    # Axes counter
    ax_counter1 = 0
    ax_counter2 = 0
    ax_counter3 = 0

    # Go to experiment folder
    for param1 in [1, 2]:
        for param2 in [1, 2]:
            for param3 in [1, 2]:
                for param4 in [1, 2, 3]:
                    path = exp_path / (
                        "a" + str(param1) + str(param2) + str(param3) + str(param4)
                    )

                    # Load stats file
                    with open(path / "stats.pickle", "rb") as file:
                        stats = pickle.load(file)

                    # Figure 1
                    if ax_counter1 < 8:

                        # Plot loss with twin axes
                        axs1[ax_counter1].plot(
                            range(1, 51),
                            _smooth(stats["list_loss_recon_val_epoch"], 0.8),
                            color="b",
                        )
                        axs1[ax_counter1].tick_params(axis="y", labelcolor="b")
                        axs1[ax_counter1].yaxis.set_major_locator(
                            MaxNLocator(nbins=5, integer=True)
                        )
                        ax2 = axs1[ax_counter1].twinx()
                        ax2.plot(
                            range(1, 51),
                            _smooth(stats["list_loss_kl_val_epoch"], 0.8),
                            color="g",
                        )
                        ax2.tick_params(axis="y", labelcolor="g")
                        ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
                        ax_counter1 += 1

                    # Figure 2
                    else:

                        if ax_counter2 < 8:

                            # Plot loss with twin axes
                            axs2[ax_counter2].plot(
                                range(1, 51),
                                _smooth(stats["list_loss_recon_val_epoch"], 0.8),
                                color="b",
                            )
                            axs2[ax_counter2].tick_params(axis="y", labelcolor="b")
                            axs2[ax_counter2].yaxis.set_major_locator(
                                MaxNLocator(nbins=5, integer=True)
                            )
                            ax2 = axs2[ax_counter2].twinx()
                            ax2.plot(
                                range(1, 51),
                                _smooth(stats["list_loss_kl_val_epoch"], 0.8),
                                color="g",
                            )
                            ax2.tick_params(axis="y", labelcolor="g")
                            ax2.yaxis.set_major_locator(
                                MaxNLocator(nbins=5, integer=True)
                            )
                            ax_counter2 += 1

                        else:

                            # Plot loss with twin axes
                            axs3[ax_counter3].plot(
                                range(1, 51),
                                _smooth(stats["list_loss_recon_val_epoch"], 0.8),
                                color="b",
                            )
                            axs3[ax_counter3].tick_params(axis="y", labelcolor="b")
                            axs3[ax_counter3].yaxis.set_major_locator(
                                MaxNLocator(nbins=5, integer=True)
                            )
                            ax2 = axs3[ax_counter3].twinx()
                            ax2.plot(
                                range(1, 51),
                                _smooth(stats["list_loss_kl_val_epoch"], 0.8),
                                color="g",
                            )
                            ax2.tick_params(axis="y", labelcolor="g")
                            ax2.yaxis.set_major_locator(
                                MaxNLocator(nbins=5, integer=True)
                            )
                            ax_counter3 += 1

    fig1.savefig(file_path1, dpi=600)
    fig2.savefig(file_path2, dpi=600)
    fig3.savefig(file_path3, dpi=600)


def _smooth(series, weight):
    last = series[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in series:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value
    return smoothed
