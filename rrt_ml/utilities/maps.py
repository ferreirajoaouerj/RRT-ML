from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

from rrt_ml.utilities.configs import *
from rrt_ml.utilities.hints import *
from rrt_ml.utilities.infos import *
from rrt_ml.utilities.paths import *


class Node:
    """
    Node (or pose) for the RRT tree.
    """

    def __init__(
            self,
            pose: Vector3,
            parent: None or 'Node' = None,
            cost_from_parent: None or 'Node' = None,
            cost_from_root: None | float = None,
            origin: None | str = None,
    ):
        """
        Initialize.
        :param pose: vector [x y theta]
        :param parent: parent pose
        :param cost_from_parent: cost of moving from parent to this pose
        :param cost_from_root: cost of moving from root node to this pose
        :param origin: whether the node comes from sampler, random, etc
        """

        self.pose = pose  # type: Vector3
        self.parent = parent  # type: None | Node
        self.cost_from_parent = cost_from_parent  # type: None | float
        self.cost_from_root = cost_from_root  # type: None | float
        self.origin = origin  # type: None | str

        self.children = []  # type: list[Node, ...]
        self.info = None  # type: None | NodeReachInfo

        self._draw_cost_from_root = False
        self._draw_cost_from_parent = False
        self._draw_color = 'black'
        self._draw_info = False
        self._draw_temp_distance = False
        self._draw_temp_info = False
        self._draw_temp_cost_from_root = False
        self._mpl_txts = []
        self._mpl_patches = []
        self._mpl_lines = []
        self._temp_distance = None
        self._temp_info = None
        self._temp_cost_from_root = None

    def set_cost_change_propagation(self):
        """
        When the 'cost_from_root' changes we need to update the 'cost_from_root' of all children.
        """

        for node_child in self.children:
            node_child.cost_from_root = (
                    self.cost_from_root + node_child.cost_from_parent
            )
            node_child.set_cost_change_propagation()

    def set_remove_me_as_child(self):
        """
        This node may belong to a list of children, and we may want to remove it.
        """

        if self.parent is not None:
            self.parent.children.remove(self)

    def set_info(self, node_reach_info: 'NodeReachInfo'):
        """
        Add stats on how this node was reached from parent node.
        """

        # Get copy of bullet stats
        self.info = node_reach_info

    def __repr__(self):
        """
        Print node.
        """

        if self.parent is not None:
            return (
                f"Pose: {self.pose}\n"
                f"Parent: {self.parent.pose}\n"
                f"Cost from parent: {self.cost_from_parent}\n"
                f"Cost from root: {self.cost_from_root}\n\n"
            )
        else:
            return f"Pose: {self.pose}\n" \
                   f"Cost from root: {self.cost_from_root}\n\n"


class Map:

    def __init__(self, cfg: MasterConfig):

        self.cfg = cfg
        self.temp_arrows = []
        self.temp_lines = []
        self.temp_texts = []
        self.nodes = []
        self.vertices = []
        self.orientations = []

        self._setup()

    def set_add_nodes(self, nodes: Iterable['Node'] | 'Node'):
        """
        Add node(s) to list of nodes.
        :param nodes: node(s) to add.
        """

        if isinstance(nodes, Iterable):
            for node in nodes:
                self.nodes.append(node)
        else:
            self.nodes.append(nodes)

    def set_add_states(self, states: Iterable[Vector4] | Vector4, origin: str = 'sl'):
        """
        Add states to list of nodes
        :param states: list of states to add
        :param origin: where the states come from
        """

        # Handle single state case
        if isinstance(states[0], Iterable):
            pass
        else:
            states = [states]

        # Iterate and add to map's list of nodes
        for st in states:
            pose = [st[0], st[1], np.arctan2(st[2], st[3])]
            self.nodes.append(Node(pose, None, None, None, origin))

    def get_plot_tree(self, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes] | plt.Axes:
        """
        Plot map with obstacles and nodes/connections as a tree.
        :return: figure and axes.
        """

        # Handle cases where axis is not given and must return a new figure
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Set up axis layout
        self._set_plot_layout(ax)

        # Add obstacles
        self._set_plot_obstacles(ax)

        # Add branches
        self._set_plot_branches(ax)
        
        if fig is None:
            return ax
        
        else:
            return fig, ax

    def get_plot_lines(self, ax: plt.Axes | None = None) -> tuple[plt.Figure, plt.Axes] | plt.Axes:
        """
        Plot map with nodes and line connections.
        :return: figure and axes or axes
        """

        # Handle cases where axis is not given and must return a new figure
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Set up axis layout
        self._set_plot_layout(ax)

        # Add obstacles
        self._set_plot_obstacles(ax)

        # Add nodes
        self._set_plot_nodes(ax)

        # Add connections
        self._set_plot_connections_lines(ax)
        
        if fig is None:
            return ax

        else:
            return fig, ax

    def set_add_temp_line(self, ax: plt.Axes, node: Node, color: str):
        """
        Add temporary node to the tree (for debugging)
        """

        self._setup_ignore_static_warnings()

        # Plot line
        line = ax.plot(node.info.xs, node.info.ys, '--', color=color)

        # Add arrow to list of temp arrows
        self.temp_lines.append(line)

    def set_add_temp_arrows(self, ax: plt.Axes, nodes: Node | list[Node, ...], color: str):
        """
        Add temporary node to the tree (for debugging)
        """

        self._setup_ignore_static_warnings()
        if not isinstance(nodes, Iterable):
            nodes = [nodes]

        # Set params for different types of nodes
        kw = dict(width=0.01, head_width=0.15, head_length=0.15, ec=color, fc=color)

        for node in nodes:
            # Get arrow params
            x, y = node.pose[0:2]
            dx, dy = 0.45 * np.cos(node.pose[2]), 0.45 * np.sin(node.pose[2])

            # Plot arrow
            arrow = ax.arrow(x, y, dx, dy, **kw)

            # Add arrow to list of temp arrows
            self.temp_arrows.append(arrow)

    def set_add_temp_text(self, ax: plt.Axes, pose, text, color):
        """
        Add temporary text to the tree (for debugging)
        """

        self._setup_ignore_static_warnings()

        # Plot text
        xp, yp = pose[:2]
        # xt, yt = pose[0] + 1, pose[1] + 1
        txt = ax.annotate(text=text, xy=(xp, yp), xycoords='data', color=color)
        # txt = ax.annotate(text=text, xy=(xp, yp), xycoords='data', xytext=(xt, yt), textcoords='data')

        # Add arrow to list of temp arrows
        self.temp_texts.append(txt)

    def set_clear_temp_objs(self):
        """
        Clear matplotlib temporary objects.
        """

        for line in self.temp_lines:
            line[0].remove()
        for arrow in self.temp_arrows:
            arrow.remove()
        for text in self.temp_texts:
            text.remove()
        self.temp_lines, self.temp_arrows, self.temp_texts = [], [], []

    def _set_plot_branches(self, ax: plt.Axes):
        """
        Add nodes and connections as branches.
        :param ax: axes to plot
        """

        # Plot connection for all nodes
        for node in self.nodes:  # type: Node

            # Initial node has no parent
            if node.parent is not None:

                # Now we take (x, y) pairs, i.e. the trajectory, from stats
                xs = node.info.xs
                ys = node.info.ys

                # Plot on ax
                ax.plot(xs, ys, color='grey', alpha=0.5)

    def _set_plot_branches_for_list(self, ax: plt.Axes, list_nodes: list['Node', ...]):
        """
        Plot branches for a list of nodes only.
        :param list_nodes: list of nodes to plot branches.
        """

        # Ignore warnings
        self._setup_ignore_static_warnings()

        # Loop provided nodes
        for node in list_nodes:
            
            # Initial node has no parent
            if node.parent is not None:

                # Now we take (x, y) pairs, i.e. the trajectory, from stats
                xs = node.info.xs
                ys = node.info.ys

                # Plot on ax
                ax.plot(xs, ys, color='blue')

    def _set_plot_connections_lines(self, ax: plt.Axes):
        """
        Add connections between nodes.
        :param ax: axes do draw connections
        """

        for node in self.nodes:

            # If node has no parent continue
            if node.parent is None:
                continue

            # Get positions
            x_i, y_i, orn_i = node.pose
            x_f, y_f, orn_f = node.parent.pose

            # Add a line from root to destination point
            ax.plot([x_i, x_f], [y_i, y_f], color="k", linewidth=1.5, alpha=0.2)

    def _set_plot_layout(self, ax: plt.Axes):
        """
        Adjust aspect ratio, grid, etc.
        :param ax: axis to change.
        """

        # Make limits according to map size
        match self.cfg.maps.general.map_name:
            case 'narrow':
                # self.cfg.maps.narrow.size
                ax.set_xlim(-0.5, 0.5 + self.cfg.maps.narrow.size)
                ax.set_ylim(-0.5, 0.5 + self.cfg.maps.narrow.size)
            case _:
                pass

        # Keep figure height and width equal
        ax.set_aspect("equal", adjustable="box")

        # Tick integer values only
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        self._setup_ignore_static_warnings()

        return ax

    def _set_plot_nodes(self, ax: plt.Axes, nodes: list[Node, ...] | None = None):
        """
        Add nodes to plot.
        :param ax: axes to plot.
        """

        # Set list of nodes to plot
        if nodes is None:
            nodes = self.nodes

        # Set params for different types of nodes
        kw_init = dict(width=0.03, head_width=0.3, head_length=0.3, ec="g", fc="g", alpha=1)
        kw_final = dict(width=0.03, head_width=0.3, head_length=0.3, ec="r", fc="r", alpha=1)
        kw_final_new = dict(width=0.01, head_width=0.15, head_length=0.15, ec="r", fc="r")
        kw_sl = dict(width=0.01, head_width=0.15, head_length=0.15, ec="k", fc="k")
        kw_random = dict(width=0.01, head_width=0.15, head_length=0.15, ec="k", fc="k")
        origin_kwargs = dict(init=kw_init, final=kw_final, final_new=kw_final_new, sl=kw_sl, random=kw_random)

        # Iterate over nodes
        for node in nodes:

            # Get arrow params
            x, y = node.pose[0:2]
            dx, dy = 0.45*np.cos(node.pose[2]), 0.45*np.sin(node.pose[2])

            # Set higher z-order for init and final node
            zorder = 3 if node.origin == 'final' or node.origin == 'init' else 1

            # Plot arrow
            ax.arrow(x, y, dx, dy, **origin_kwargs[node.origin], zorder=zorder)

    def _set_plot_obstacles(self, ax: plt.Axes):
        """
        Get rectangles patches.
        :param ax: axes to plot.
        """

        patches = []
        for v, o in zip(self.vertices, self.orientations):
            # Get params
            width = v[2] - v[0]
            height = v[1] - v[3]
            x = v[0]
            y = v[1] - height

            # Create patch and add to list
            rect = Rectangle((x, y), width, height, o)
            patches.append(rect)

        # Create black patch collection
        patches = PatchCollection(patches)
        patches.set_color("black")

        # Add to plot
        ax.add_collection(patches)

    def _set_wall(self, top_left_bottom_right: Vector4, space_snwe: Vector4 = (0, 0, 0, 0)):
        """
        Set a wall to map as a rectangle, making it shorter to not collide with other walls
        :param top_left_bottom_right: top left and bottom right vertices positions
        :param space_snwe: cut this much from each side of the rectangle (south-north-west-east)
        """

        # TopLeft and BottomRight aliases
        tl = top_left_bottom_right[:2]
        br = top_left_bottom_right[2:]

        # SpaceNorth, SpaceSouth, SpaceWest and SpaceEast aliases
        ss, sn, sw, se = space_snwe

        # Move top left vertex
        tl = [tl[0] + sw, tl[1] - sn]

        # Move bottom right vertex
        br = [br[0] - se, br[1] + ss]

        self.vertices.append([*tl, *br])
        self.orientations.append(0)

    def _setup(self):
        """
        Set up.
        """

        match self.cfg.maps.general.map_name:
            case 'arena':
                self._setup_arena()
            case 'narrow':
                self._setup_narrow()
            case _:
                raise NotImplementedError

    def _setup_arena(self, length=None, width=None, offset=None):
        """
        Set a rectangular arena.
        :param length: length of arena
        :param width: width of arena
        :param offset: offset center of arena
        """

        # Config constants
        if length is None:
            length = self.cfg.maps.arena.length
        if width is None:
            width = self.cfg.maps.arena.width
        if offset is None:
            offset = self.cfg.maps.arena.offset
        ox, oy = offset

        # Fine tune constants: WallThickness, Spacing
        wt = 0.3
        s = 1e-3

        # Make vertices
        left_wall_pos = [-wt + ox, width + oy, ox, oy]
        right_wall_pos = [length + ox, width + oy, length + wt + ox, oy]
        bottom_wall_pos = [ox, oy, length + ox, -wt + oy]
        top_wall_pos = [ox, width + wt + oy, length + ox, width]

        # Set walls with spacing
        self._set_wall(bottom_wall_pos, space_snwe=[0, s, 0, 0])
        self._set_wall(top_wall_pos, space_snwe=[s, 0, 0, 0])
        self._set_wall(left_wall_pos, space_snwe=[0, 0, 0, s])
        self._set_wall(right_wall_pos, space_snwe=[0, 0, s, 0])

    def _setup_narrow(self):
        """
        Set up narrow map.
        """

        # Config constants
        size = self.cfg.maps.narrow.size
        x1, y1 = self.cfg.maps.narrow.narrow1_pos
        x2, y2 = self.cfg.maps.narrow.narrow2_pos
        ww = self.cfg.maps.narrow.wall_width
        pw = self.cfg.maps.narrow.passage_width

        # Fine tune constants: Spacing
        s = 1e-3

        # Box around map
        self._setup_arena(size, size, offset=(0, 0))

        # Left lower, left upper, right lower and right upper walls
        ll = [x1 - ww / 2, y1 - pw / 2, x1 + ww / 2, 0]
        lu = [x1 - ww / 2, size, x1 + ww / 2, y1 + pw / 2]
        rl = [x2 - ww / 2, y2 - pw / 2, x2 + ww / 2, 0]
        ru = [x2 - ww / 2, size, x2 + ww / 2, y2 + pw / 2]

        # Set walls
        self._set_wall(ll, space_snwe=[s, 0, 0, 0])
        self._set_wall(lu, space_snwe=[0, s, 0, 0])
        self._set_wall(rl, space_snwe=[s, 0, 0, 0])
        self._set_wall(ru, space_snwe=[0, s, 0, 0])

    def _setup_ignore_static_warnings(self):
        """
        Ignore static warnings.
        """
        pass

