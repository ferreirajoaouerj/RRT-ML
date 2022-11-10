import math
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pybullet as p
import seaborn as sns
import yaml
from matplotlib import pyplot as plt

from rrt_ml.algorithms.rl import *
from rrt_ml.algorithms.rrt import *
from rrt_ml.algorithms.sl import *
from rrt_ml.utilities.analytic import *
from rrt_ml.utilities.configs import *
from rrt_ml.utilities.maps import *
from rrt_ml.utilities.paths import *


class Hyper:

    def __init__(self, cfg: MasterConfig):
        """
        Initialize.
        :param cfg: general config
        """

        self.cfg = cfg
        self.cfgs = None  # type: None | list[MasterConfig, ...]
        self.algorithm = None  # type: RL | SL | RRT | None
        self.path = None  # type: None | Path
        self.prefix = None  # type: None | str
        self.hyper_dict = None  # type: None | dict

        self._setup()

    def train(self):
        """
        Train all hyperparameters.
        """

        # Test for different configs
        for i, cfg in enumerate(self.cfgs):

            # Don't load if next config experiment folder exists
            # if i != len(self.cfgs) - 1:
            #     next_cfg_name = self.cfgs[i+1].general.config_name_or_prefix
            #     next_cfg_exp_path = Paths().exp('rrt', next_cfg_name)
            #     if next_cfg_exp_path.exists():
            #         continue

            # Train
            alg = self.algorithm(cfg)
            alg.train()

    def test(self):
        """
        Get various results.
        """

        match self.cfg.general.algorithm:
            case 'rl':
                self._test_rl()
            case 'rrt':
                self._test_rrt()
            case 'sl':
                self._test_sl()
            case _:
                raise NotImplementedError

    def _test_rl(self):
        """
        Test RL configs.
        """

        self._test_rl_plot_rewards_entropy()

    def _test_rl_plot_rewards_entropy(self):
        """
        Plot rewards for all configs
        """

        # Dict to hold data
        l_stats = []
        l_names = []

        # Loop configs
        for cfg in self.cfgs:

            # Load RL, get stats and name
            l_stats.append(RL(cfg).stats)
            l_names.append(cfg.general.config_name_or_prefix)

        # Figures
        fig, axs = plt.subplots(2, 2, sharey='row', sharex='col', figsize=(9, 6))
        fig.subplots_adjust(wspace=0.15, hspace=0.2)
        axs = axs.flatten()

        for i, ax in enumerate(axs):

            # Plot reward
            ax.plot(l_stats[i].train_timestep_nums, l_stats[i].total_val_rewards, label='Recompensa', color='blue')

            # Set tick color to blue
            ax.tick_params(axis='y', labelcolor='blue')

            # Get entropy
            csv_path = next(Paths().exp('rl', l_names[0]).glob('*entropy*'))
            df = pd.read_csv(str(csv_path))

            # Plot
            axt = ax.twinx()
            axt.plot(df['step'][7:], df['Explore/ag_kde_entropy'][7:], label='Entropia', color='green')

            # Set color to green
            axt.tick_params(axis='y', labelcolor='green')

            # Legends
            ax.legend(loc='lower right')
            axt.legend(bbox_to_anchor=(1, 0.31))

            # Grid
            ax.grid(True)

            # X axis label
            if i in [2, 3]:
                ax.set_xlabel('Passos de treinamento')

        fig_name = f'rlHiperRecompensaEntropia.png'
        fig.savefig(str(self.path / fig_name), dpi=600, bbox_inches='tight')

    def _test_rrt(self):
        """
        Test RRT configs.
        """

        # Generate results for CVAE experiment
        if 'cvae_rs' in self.cfg.general.config_name_or_prefix:
            self._test_rrt_compare_time_to_solve_cvae()

        # Generate all other results
        df = pd.DataFrame({
            'Critério': [],
            'Custo': [],
            'Tempo de resolução (s)': [],
            'Tempo de inferência': [],
            'Sucesso': [],
            'Semente': [],
            'RRT*': []
        })

        # Loop over all configs to generate test results
        loaded_df = False
        for cfg in self.cfgs:

            # Break if dataframe exists
            if (self.path / 'results_dataframe.csv').exists():
                df = pd.read_csv(str(self.path / 'results_dataframe.csv'))
                loaded_df = True
                break

            # Test function will generate simulation info attribute and figures
            rrt = RRT(cfg)
            rrt.test()

            # Decide what is the RRT* variation
            match cfg.rrt.general.controller:
                case 'rl':
                    rrt_var = 'ApM'
                case _:
                    rrt_var = 'RS+MPC'

            # Add row for both criterion
            time_to_solve = rrt._test_get_time_to_solve()
            df_row1 = pd.DataFrame({
                'Critério': ['Tempo (s)'],
                'Custo': [rrt._test_get_solution_simulation_cost_time()],
                'Tempo de resolução (s)': [time_to_solve],
                'Tempo de inferência': [rrt.solution_simulation_info.cpu_time],
                'Sucesso': [rrt.solution_simulation_info.success],
                'Semente': [cfg.rrt.general.seed],
                'RRT*': [rrt_var],
            })
            df_row2 = pd.DataFrame({
                'Critério': ['Distância (m)'],
                'Custo': [rrt._test_get_solution_simulation_cost_length()],
                'Tempo de resolução (s)': [time_to_solve],
                'Tempo de inferência': [rrt.solution_simulation_info.cpu_time],
                'Sucesso': [rrt.solution_simulation_info.success],
                'Semente': [cfg.rrt.general.seed],
                'RRT*': [rrt_var],
            })

            # Add row to main dataframe
            df = pd.concat((df, df_row1, df_row2), ignore_index=True)

            # Top view of solution
            self._test_rrt_solution_top_view(cfg, rrt)

            # Tree of solution
            self._test_rrt_solution_tree(cfg, rrt)

            # Disconnect gui env
            p.disconnect(rrt.env.client)

        # Save dataframe
        if not loaded_df:
            # noinspection PyTypeChecker
            df.to_csv(str(self.path / 'results_dataframe.csv'), index=0)

        # Comparison of costs
        #self._test_rrt_compare_success_rates(df)
        #self._test_rrt_compare_costs(df)
        #self._test_rrt_compare_resolution_time(df)
        #self._test_rrt_compare_inference_time()
        #self._test_rrt_plot_all_trees()
        self._test_rrt_plot_all_top_view()

    def _test_rrt_compare_costs(self, df):
        """
        Compare solution costs across approaches.
        :param df: data table
        """

        # Compare mean value when success
        df_copy = df.copy(deep=True)
        seeds = np.unique(df.loc[df['Sucesso'] == 0.0, 'Semente'].values)
        for i, row in df.iterrows():
            if row['Semente'] in seeds:
                df_copy = df_copy.drop(i)

        # Save figure
        path = Paths().exp('rrt', self.prefix) / 'comparacaoCustos.png'
        sns.set_style('whitegrid')
        ax = sns.barplot(data=df_copy, x="Critério", y="Custo", hue="RRT*")
        ax.get_figure().savefig(str(path), dpi=600, bbox_inches='tight')
        plt.close()

    def _test_rrt_compare_inference_time(self):
        """
        Compare inference time for controllers.
        """

        # Start
        rrt = RRT(self.cfgs[0])
        obs = rrt.env.reset()
        rrt._get_action_rl(obs)
        mpcrs = MPCRS(self.cfgs[0])
        mpcrs.get_action(obs)

        # Check time
        a = time.time()
        for _ in range(100):
            rrt._get_action_rl(obs)
        time_rl = (time.time() - a) / 100 * 1000
        a = time.time()
        for _ in range(100):
            mpcrs.get_action(obs)
        time_mpc = (time.time() - a) / 100 * 1000

        # Get dataframe
        df = pd.DataFrame(data={'Tempo de inferência (ms)': [time_rl, time_mpc],
                                'Controlador': ['Agente', 'CPM']})

        # # Save figure
        path = Paths().exp('rrt', self.prefix) / 'tempoInferenciaControle.png'
        sns.set_style('whitegrid')
        ax = sns.barplot(data=df, x="Tempo de inferência (ms)", y="Controlador", orient='h')
        ax.get_figure().savefig(str(path), dpi=600, bbox_inches='tight')
        plt.close()

    def _test_rrt_compare_resolution_time(self, df):
        """
        Compare time to find a solution in RRT.
        :param df: data table
        """

        # Save figure
        path = Paths().exp('rrt', self.prefix) / 'custoComputacionalRrtEstrela.png'
        sns.set_style('whitegrid')
        ax = sns.barplot(data=df, x="RRT*", y="Tempo de resolução (s)")
        ax.get_figure().savefig(str(path), dpi=600, bbox_inches='tight')
        plt.close()

    def _test_rrt_compare_success_rates(self, df):
        """
        Compare success rate for the controllers.
        :param df: data table
        """

        # Save figure
        path = Paths().exp('rrt', self.prefix) / 'rrtTaxaSucessoControladores.png'
        sns.set_style('whitegrid')
        ax = sns.barplot(x='RRT*', y='Sucesso', data=df, estimator=lambda x: sum(x == 1) * 100.0 / len(x))
        ax.set(ylabel='Taxa de sucesso (%)')
        ax.get_figure().savefig(str(path), dpi=600, bbox_inches='tight')
        plt.close()

    def _test_rrt_compare_time_to_solve_cvae(self):
        """
        Compare resolution time and costs when using cvae.
        """

        path = Paths().exp('rrt', self.prefix) / 'tempoResolucaoAlpha.png'
        if path.exists():
            return

        times_cvae = []
        times_uniform = []
        costs_cvae = []
        costs_uniform = []
        for cfg in self.cfgs:

            # Initialize algo
            rrt = RRT(cfg)

            # Check config
            match cfg.rrt.sample.learner_prob:
                case 0:
                    times_uniform.append(rrt._test_get_time_to_solve())
                    costs_uniform.append(rrt.node_final.cost_from_root)
                case _:
                    times_cvae.append(rrt._test_get_time_to_solve())
                    costs_cvae.append(rrt.node_final.cost_from_root)

            # Disconnect env
            p.disconnect(physicsClientId=rrt.env.client)

        # Get mean values
        time_cvae = np.mean(times_cvae)
        time_uniform = np.mean(times_uniform)
        cost_cvae = np.mean(costs_cvae)
        cost_uniform = np.mean(costs_uniform)

        # Normalize to 0-1
        prop_time = time_cvae / time_uniform
        prop_cost = cost_cvae / cost_uniform

        # Plot with seaborn
        df = pd.DataFrame([
            ['Tempo de resolução (s)', 1, r'$\alpha=0$'],
            ['Tempo de resolução (s)', prop_time, r'$\alpha=0.7$'],
            ['Custo (m)', 1, r'$\alpha=0$'],
            ['Custo (m)', prop_cost, r'$\alpha=0.7$'],
        ], columns=['Critério', 'Valor', r'$\alpha$'])

        sns.set(font_scale=1.2)
        sns.set_style('whitegrid')
        ax = sns.barplot(data=df, x="Critério", y="Valor", hue='$\\alpha$')
        ax.set(yticklabels=[])  # remove the tick labels
        ax.set(ylabel='Valores normalizados')
        sns.move_legend(ax, 'upper center')
        ax.get_figure().savefig(str(path), dpi=600)
        plt.close()

    def _test_rrt_plot_all_top_view(self):
        """
        Plot and compare all trees and solutions for all problems.
        """

        # Get path
        path = Paths().exp('rrt', self.prefix)
        path1 = path / 'topViewTodas1.png'
        path2 = path / 'topViewTodas2.png'
        # if path1.exists() and path2.exists():
        #     return

        # Create figures
        figs, l_axs = [], []
        for i in range(2):
            fig, axs = plt.subplots(3, 3, sharey='row', sharex='col', figsize=(12, 9.6))
            fig.subplots_adjust(wspace=-0.4, hspace=0.3)
            figs.append(fig)
            l_axs.append(axs.flatten())
        all_axs = np.concatenate(l_axs)

        # Loop axes
        for i, (fig, axs) in enumerate(zip(figs, l_axs)):
            for j, ax in enumerate(axs):

                # Initialize algo and map
                rrt = RRT(self.cfgs[i * 9 + j])

                # Remove car and target from origin
                rrt.env.car.set_pose([-10, -10, 0])
                rrt.env.target.set_pose([-10, -10, 0])

                # Place ghosts, get image and then remove ghosts
                rrt.env._set_place_trajectory_ghosts(episode_num=0, stats=rrt.solution_simulation_info.ep_stats)
                img = rrt.env._get_image_current_top_view()
                rrt.env._set_remove_ghosts()
                ax.imshow(img, origin='lower', extent=rrt.cfg.env.test.narrow.img_extent)

                # Check which algorithm is running
                alg = 'CPM' if rrt.cfg.rrt.general.controller == 'mpc' else 'Agente'

                # Axes labels
                if j in [0, 3, 6]:
                    ax.set_ylabel('y [m]')
                if j in [6, 7, 8]:
                    ax.set_xlabel('x [m]')

                # Title
                ax.set_title(f'Problema {j+1} - {alg}')

                # Disconnect test env
                p.disconnect(physicsClientId=rrt.env.client)

        # Save figure
        figs[0].savefig(path1, dpi=600, bbox_inches='tight', pad_inches=0.1)
        figs[1].savefig(path2, dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def _test_rrt_plot_all_trees(self):
        """
        Plot and compare all trees and solutions for all problems.
        """

        # Get path
        path = Paths().exp('rrt', self.prefix)
        path1 = path / 'arvoreSolucaoTodas1.png'
        path2 = path / 'arvoreSolucaoTodas2.png'
        if path1.exists() and path2.exists():
            return

        # Create figures
        figs, l_axs = [], []
        for i in range(2):
            fig, axs = plt.subplots(3, 3, sharey='row', sharex='col', figsize=(12, 9.6))
            fig.subplots_adjust(wspace=-0.4, hspace=0.3)
            figs.append(fig)
            l_axs.append(axs.flatten())
        all_axs = np.concatenate(l_axs)

        # Loop axes
        for i, (fig, axs) in enumerate(zip(figs, l_axs)):
            for j, ax in enumerate(axs):

                # Initialize algo and map
                rrt = RRT(self.cfgs[i*9 + j])
                m = Map(rrt.cfg)

                # Add normal nodes
                m.set_add_nodes(rrt.list_nodes)
                m.get_plot_tree(ax)

                # Add solution
                m._set_plot_branches_for_list(ax, list_nodes=rrt.solution_simulation_info.list_nodes)

                # Add initial and final nodes
                nodes = [rrt.solution_simulation_info.list_nodes[0], rrt.solution_simulation_info.list_nodes[-1]]
                m._set_plot_nodes(ax, nodes=nodes)

                # Axes labels
                if j in [0, 3, 6]:
                    ax.set_ylabel('y [m]')
                if j in [6, 7, 8]:
                    ax.set_xlabel('x [m]')

                # Title
                ax.set_title(f'Problema {j+1}')

                # Disconnect test env
                p.disconnect(physicsClientId=rrt.env.client)

        # Save figure
        figs[0].savefig(path1, dpi=600, bbox_inches='tight', pad_inches=0.1)
        figs[1].savefig(path2, dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def _test_rrt_solution_top_view(self, cfg, rrt):
        """
        Get top view of solution trajectory
        :param cfg: current config
        :param rrt: RRT* instance
        """

        # Get path
        path = Paths().exp('rrt', self.prefix) / ('solucaoTopView' + '_' + cfg.general.config_name_or_prefix + '.png')
        if path.exists():
            return

        # Place ghosts, get image and then remove ghosts
        rrt.env._set_place_trajectory_ghosts(episode_num=0, stats=rrt.solution_simulation_info.ep_stats)
        img = rrt.env._get_image_current_top_view()
        rrt.env._set_remove_ghosts()

        # Plot on axis
        fig, ax = plt.subplots()
        ax.imshow(img, origin='lower', extent=cfg.env.test.narrow.img_extent)
        fig.savefig(path, dpi=600, bbox_inches='tight')
        plt.close()

    def _test_rrt_solution_tree(self, cfg, rrt):
        """
        Get tree view of solution trajectory
        :param cfg: current config
        :param rrt: RRT* instance
        """

        # Get path
        path = Paths().exp('rrt', self.prefix)
        path = path / ('arvoreSolucaoTopView' + '_' + cfg.general.config_name_or_prefix + '.png')
        if path.exists():
            return

        # Make tree
        mmap = Map(cfg)
        mmap.set_add_nodes(rrt.list_nodes)
        fig, ax = mmap.get_plot_tree()

        # Make solution line
        mmap._set_plot_branches_for_list(ax, list_nodes=rrt.solution_simulation_info.list_nodes)

        # Plot initial and final nodes
        nodes = [rrt.solution_simulation_info.list_nodes[0], rrt.solution_simulation_info.list_nodes[-1]]
        mmap._set_plot_nodes(ax, nodes=nodes)

        # Axis info
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

        # Save figure
        fig.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def _test_sl(self):
        """
        Test SL configs.
        """

        # self._test_sl_compare_samples_all()
        # self._test_sl_compare_samples_eight()
        # self._test_sl_compare_samples_two()
        self._test_sl_compare_error_three()

    def _test_sl_compare_samples_all(self):
        """
        Compare samples of all configs.
        """

        # Constants
        n_samples = 15
        seed = 0
        map_nums = [0, 20, 40, 60, 80, 100, 120]

        # Loop over maps
        for map_num in map_nums:

            # Return if figure exists
            name = f'sl_hyper_comparacao_amostras_todos_mapa_{map_num}_fig_0.png'
            if (self.path / name).exists():
                continue

            # Constants
            n_rows = 4
            n_cols = 4
            n_cfgs = len(self.cfgs)
            n_figs = math.ceil(n_cfgs / (n_rows * n_cols))

            # Create figures, flatten list of axes before adding to list and make a big list of axes
            figs, l_axs = [], []
            for i in range(n_figs):
                fig, axs = plt.subplots(n_rows, n_cols, sharey='row', sharex='col', figsize=(12, 9.6))
                fig.subplots_adjust(wspace=-0.4, hspace=0.3)
                figs.append(fig)
                l_axs.append(axs.flatten())
            all_axs = np.concatenate(l_axs)

            # Get conditions
            obstacles, state_i, state_f = SL(self.cfgs[0]).get_conditions(map_num)

            # Loop through configs and axes
            for i, (cfg, ax) in enumerate(zip(self.cfgs, all_axs)):

                # Get states
                states = SL(cfg).get_samples(map_num, n_samples, np.random.default_rng(seed))

                # Update config before drawing map (obstacles plot comes from config)
                cfg.maps.narrow.narrow1_pos = obstacles[0:2]
                cfg.maps.narrow.narrow2_pos = obstacles[2:4]

                # Initialize map and add states
                mapp = Map(cfg)
                mapp.set_add_states(states, 'sl')
                mapp.set_add_states(state_i, 'init')
                mapp.set_add_states(state_f, 'final')

                # Add plot to current axis
                mapp.get_plot_lines(ax)

                # Axes labels
                if i % n_cols == 0:
                    ax.set_ylabel('y [m]')
                if (i % (n_rows * n_cols)) >= (n_cols * n_rows - n_cols):
                    ax.set_xlabel('x [m]')

                # Title
                ax.set_title(cfg.general.config_name_or_prefix)

            # Save figures
            for i, fig in enumerate(figs):
                fig_name = f'sl_hyper_comparacao_amostras_todos_mapa_{map_num}_fig_{i}.png'
                fig.savefig(str(self.path / fig_name), dpi=600, bbox_inches='tight')

    def _test_sl_compare_samples_eight(self):
        """
        Compare two configs on 4 maps.
        """

        # Return if figure exists
        path = self.path / 'sl_hyper_comparacao_amostras_oito.png'
        if path.exists():
            return

        # Constants
        # cfg_names = ['h_2421', 'h_2421', 'h_2421', 'h_2421', 'h_2521', 'h_2521', 'h_2521', 'h_2521']
        # map_nums = [5, 6, 7, 8, 5, 6, 7, 8]
        cfg_names = ['h_2121', 'h_2221', 'h_2321', 'h_2421', 'h_2521', 'h_2621', 'h_2721', 'h_2821']
        map_nums = [5, 5, 5, 5, 5, 5, 5, 5]
        n_samples = 15
        seed = 0

        # Constants
        n_rows = 2
        n_cols = 4

        # Create figure
        fig, axs = plt.subplots(n_rows, n_cols, sharey='row', sharex='col', figsize=(12, 9.6))
        fig.subplots_adjust(hspace=-0.45, wspace=0.15)
        axs = axs.flatten()

        # Loop through configs and axes
        for i, (cfg_name, ax, map_num) in enumerate(zip(cfg_names, axs, map_nums)):

            # Get config and SL
            cfg = self._get_config(cfg_name)
            sl = SL(cfg)

            # Get conditions, states and delete SL
            obstacles, state_i, state_f = sl.get_conditions(map_num)
            states = sl.get_samples(map_num, n_samples, np.random.default_rng(seed))
            del sl

            # Update config before drawing map (obstacles plot comes from config)
            cfg.maps.narrow.narrow1_pos = obstacles[0:2]
            cfg.maps.narrow.narrow2_pos = obstacles[2:4]

            # Initialize map and add states
            mapp = Map(cfg)
            mapp.set_add_states(states, 'sl')
            mapp.set_add_states(state_i, 'init')
            mapp.set_add_states(state_f, 'final')

            # Add plot to current axis
            mapp.get_plot_lines(ax)

            # Axes labels
            if i % n_cols == 0:
                ax.set_ylabel('y [m]')
            if (i % (n_rows * n_cols)) >= (n_cols * n_rows - n_cols):
                ax.set_xlabel('x [m]')

            # Title
            ax.set_title(cfg.general.config_name_or_prefix + ' (' + r'$\beta=$' + str(cfg.sl.loss.beta) + ')', )

        # Save figure
        fig.savefig(str(path), dpi=600, bbox_inches='tight')

    def _test_sl_compare_samples_two(self):
        """
        Compare state generation of two configs.
        """

        # Return if figure exists
        path = self.path / 'sl_hyper_comparacao_amostras_dois.png'
        if path.exists():
            return

        # Constants
        cfg1_name, cfg2_name = ['h_2421', 'h_2222']
        map_num = 1
        n_samples = 15
        seed = 0

        # Get the two configs to compare
        cfg1, cfg2 = None, None
        for cfg in self.cfgs:
            cfg1 = cfg if cfg.general.config_name_or_prefix == cfg1_name else cfg1
            cfg2 = cfg if cfg.general.config_name_or_prefix == cfg2_name else cfg2

        # Get states and conditions
        states1 = SL(cfg1).get_samples(map_num, n_samples, np.random.default_rng(seed))
        states2 = SL(cfg2).get_samples(map_num, n_samples, np.random.default_rng(seed))
        obstacles, state_i, state_f = SL(cfg1).get_conditions(map_num)

        # Update config with obstacles
        cfg1.maps.narrow.narrow1_pos = obstacles[0:2]
        cfg1.maps.narrow.narrow2_pos = obstacles[2:4]
        cfg2.maps.narrow.narrow1_pos = obstacles[0:2]
        cfg2.maps.narrow.narrow2_pos = obstacles[2:4]

        # Get maps
        map1 = Map(cfg1)
        map2 = Map(cfg2)

        # Add samples, initial and final states
        map1.set_add_states(states1, 'sl')
        map1.set_add_states(state_i, 'init')
        map1.set_add_states(state_f, 'final')
        map2.set_add_states(states2, 'sl')
        map2.set_add_states(state_i, 'init')
        map2.set_add_states(state_f, 'final')

        # Plot maps
        fig, axs = plt.subplots(1, 2, sharey='all', squeeze=True, figsize=(9, 6))
        fig.subplots_adjust(hspace=-0.2)
        map1.get_plot_lines(axs[0])
        map2.get_plot_lines(axs[1])

        # Adjust axes
        axs[0].set_ylabel('y [m]')
        axs[0].set_xlabel('x [m]')
        axs[1].set_xlabel('x [m]')

        # Title
        axs[0].set_title(cfg1.general.config_name_or_prefix)
        axs[1].set_title(cfg2.general.config_name_or_prefix)

        # Save figure
        fig.savefig(str(path), dpi=600, bbox_inches='tight')

    def _test_sl_compare_error_three(self):
        """
        Plot error for three configs.
        """

        # Return if figure exists
        path = self.path / 'sl_hyper_comparacao_erro_tres.png'
        if path.exists():
            return

        # Constants
        # cfg_names = ['h_2121', 'h_2421', 'h_2821']
        cfg_names = ['h_2421', 'h_2411', 'h_2322']

        # Constants
        n_rows = 3
        n_cols = 1

        # Create figure
        fig, axs = plt.subplots(n_rows, n_cols, sharey='row', sharex='col', figsize=(12, 9.6))
        axs = axs.flatten()

        # Loop through configs and axes
        for i, (cfg_name, ax) in enumerate(zip(cfg_names, axs)):

            # Get stats
            cfg = self._get_config(cfg_name)
            stats = SL(cfg).stats

            # Plot
            ax.plot(range(1, 50), stats.val_recon_loss, label='Erro de reconstrução')
            ax2 = ax.twinx()
            ax2.plot(range(1, 50), stats.val_kl_loss, color='green', label='Erro de divergência KL')

            # Change axis color
            ax.yaxis.set_tick_params(labelcolor='blue')
            ax2.yaxis.set_tick_params(labelcolor='green')

            # Legend
            lh1, l1 = ax.get_legend_handles_labels()
            lh2, l2 = ax2.get_legend_handles_labels()
            ax.legend([lh1[0]]+[lh2[0]], [l1[0]]+[l2[0]])

            # Axes labels
            if (i % (n_rows * n_cols)) >= (n_cols * n_rows - n_cols):
                ax.set_xlabel('Época')

            # Title
            ax.set_title(cfg.general.config_name_or_prefix + ' (' + r'$\beta=$' + str(cfg.sl.loss.beta) + ')', )

        # Save figure
        fig.savefig(str(path), dpi=600, bbox_inches='tight')

    def _get_config(self, name: str):
        """
        Get config by name.
        :param name: name of config
        :return: master config
        """

        for cfg in self.cfgs:
            if cfg.general.config_name_or_prefix == name:
                return cfg.copy(deep=True)

    def _get_values_and_keys(self, dic: dict, name: str, values: list, keys: list):
        """
        Get attributes names and values.
        :param dic: hyperparameter config dict
        :param name: name of current attribute
        :param values: list of attributes values
        :param keys: list of keys names
        """

        for k, v in dic.items():
            new_name = name + '-' + k if name != "" else k
            if isinstance(v, dict):
                self._get_values_and_keys(v, new_name, values, keys)
            elif v is not None:
                values.append(v)
                keys.append(new_name)

    def _setup(self):
        """
        Set up.
        """

        self._setup_prefix()
        self._setup_alg_dict_paths()
        self._setup_save_config_to_file()
        self._setup_get_params()
        self._setup_remove_invalid_configs()

    def _setup_ignore_static_warnings(self):
        """
        Avoid static method warnings.
        """

        pass

    def _setup_prefix(self):
        """
        Set attribute that contains the name of hyperparameter search config.
        """

        self.prefix = self.cfg.general.config_name_or_prefix

    def _setup_alg_dict_paths(self):
        """
        Set up dict of params to change in the chosen algorithm.
        """

        match self.cfg.general.algorithm:
            case 'rl':
                self.algorithm = RL
                self.hyper_dict = self.cfg.hyperparams.rl.dict()
                self.path = Paths().experiments_rl / self.prefix
            case 'rrt':
                self.algorithm = RRT
                self.hyper_dict = self.cfg.hyperparams.rrt.dict()
                self.path = Paths().experiments_rrt / self.prefix
            case 'sl':
                self.algorithm = SL
                self.hyper_dict = self.cfg.hyperparams.sl.dict()
                self.path = Paths().experiments_sl / self.prefix
            case _:
                raise NotImplementedError

        # Create dir if not exist
        if not self.path.exists():
            self.path.mkdir()

    def _setup_get_params(self):
        """
        Get parameters from config.
        """

        # Get list of keys and values
        values, keys = [], []
        self._get_values_and_keys(self.hyper_dict, "", values, keys)

        # Get permutation of the indexes (suffixes)
        count = []
        for val in values:
            count.append(list(range(1, len(val) + 1)))

        # 'h_111' or 'h_010101' (if there are more than 9 values for any parameter)
        if not any([max(l) > 9 for l in count]):
            suffixes = [''.join([str(n) for n in item]) for item in product(*count)]
        else:
            suffixes = [''.join([str(n).zfill(2) for n in item]) for item in product(*count)]

        # Also permute the values
        values = list(product(*values))

        # Save values, suffixes and keys
        self.values = values
        self.suffixes = suffixes
        self.keys = keys

        # Set attributes in config and set new config name
        cfgs = []
        for value, suffix in zip(values, suffixes):

            # Create copy of original parameters
            cfg_copy = self.cfg.copy(deep=True)

            # Change experiment name
            cfg_copy.general.config_name_or_prefix = self.prefix + '_' + suffix
            cfg_copy.general.description = f"{keys} / {suffix} / {value}"

            # Change hyperparams attributes
            for attr, val in zip(keys, value):

                # Get alias to algorithm level config
                match self.cfg.general.algorithm:
                    case 'rl':
                        cfg_copy_alg = cfg_copy.rl
                    case 'rrt':
                        cfg_copy_alg = cfg_copy.rrt
                    case 'sl':
                        cfg_copy_alg = cfg_copy.sl
                    case _:
                        raise NotImplementedError

                # Get each key in nested key name to access last level of config
                key_chain = attr.split('-')

                # Access attribute
                for key in key_chain[:-1]:
                    cfg_copy_alg = cfg_copy_alg.__getattribute__(key)
                cfg_copy_alg.__setattr__(key_chain[-1], val)

            # Add to list of configs
            cfgs.append(cfg_copy)

        # Add list of configs as attribute
        self.cfgs = cfgs

    def _setup_remove_invalid_configs(self):
        """
        Remove invalid configs, for example, RRT with MPC as controller and rl distance function.
        """

        # Initialize new lists
        cfgs, values, suffixes = [], [], []

        # Loop current configs
        for i, cfg in enumerate(self.cfgs):

            # Invalid configs by algorithm
            match self.cfg.general.algorithm:

                # Invalid configs for RRT
                case 'rrt':

                    # Can't have MPC without RS
                    if cfg.rrt.general.controller == 'mpc' and cfg.rrt.general.distance_fn != 'rs':
                        continue

                    # Don't use RS with RL
                    if cfg.rrt.general.controller == 'rl' and cfg.rrt.general.distance_fn == 'rs':
                        continue

            # Add to new lists
            cfgs.append(self.cfgs[i])
            values.append(self.values[i])
            suffixes.append(self.suffixes[i])

        # Replace
        self.cfgs = cfgs
        self.values = values
        self.suffixes = suffixes

    def _setup_save_config_to_file(self):
        """
        Save config to file.
        """

        # Save config at experiment / algorithm / experiment name folder
        path = self.path / 'config.yaml'
        if not path.exists():
            with open(str(path), 'w') as f:
                yaml.dump(self.cfg.dict(), f)
