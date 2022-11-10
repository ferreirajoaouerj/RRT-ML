from matplotlib import pyplot as plt

from rrt_ml.environments.car_navigation_bullet_env import *
from rrt_ml.utilities.configs import *


# Set stats to true to track bullet stats
cfg = MasterConfig()
cfg.env.general.gui = True
cfg.maps.general.map_name = 'narrow'
cfg.maps.narrow.narrow1_pos = [3.5, 3.5]
cfg.maps.narrow.narrow2_pos = [7, 5.5]
cfg.env.car.pose = [1.5, 6.0, 1.2]
cfg.env.target.pose = [8.5, 4.0, -0.6]

# Reset master to track stats
env = CarNavigationBulletEnv(cfg)
env.reset()
img = env._get_image_current_top_view()
plt.imshow(img, origin="lower", extent=env.img_extent)
plt.savefig('narrow_top_view.png', dpi=600)