from rrt_ml.environments.car_navigation_bullet_env import *
from rrt_ml.utilities.configs import *


# Set stats to true to track bullet stats
cfg = MasterConfig()
cfg.env.general.gui = True

# Reset master to track stats
env = CarNavigationBulletEnv(cfg)
env.target.set_pose([0, 0, 0])

env.car.set_pose([0.0, 0.0, 0])
obs = env._get_observation()
print(env.compute_reward(obs['achieved_goal'], obs['desired_goal']))
img = env._get_image_current_top_view()
img.save('reward_episilon_good_1.png')

env.car.set_pose([0.01, 0.01, 0])
obs = env._get_observation()
print(env.compute_reward(obs['achieved_goal'], obs['desired_goal']))
img = env._get_image_current_top_view()
img.save('reward_episilon_good_2.png')

env.car.set_pose([0, 0, 0.002])
obs = env._get_observation()
print(env.compute_reward(obs['achieved_goal'], obs['desired_goal']))
img = env._get_image_current_top_view()
img.save('reward_episilon_good_3.png')

env.car.set_pose([0.02, 0.01, 0])
obs = env._get_observation()
print(env.compute_reward(obs['achieved_goal'], obs['desired_goal']))
img = env._get_image_current_top_view()
img.save('reward_episilon_bad_1.png')

env.car.set_pose([0.01, 0.02, 0.04])
obs = env._get_observation()
print(env.compute_reward(obs['achieved_goal'], obs['desired_goal']))
img = env._get_image_current_top_view()
img.save('reward_episilon_bad_2.png')

env.car.set_pose([0.01, 0.01, 0.06])
obs = env._get_observation()
print(env.compute_reward(obs['achieved_goal'], obs['desired_goal']))
img = env._get_image_current_top_view()
img.save('reward_episilon_bad_3.png')

pass


