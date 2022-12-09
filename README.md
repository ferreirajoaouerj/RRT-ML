# RRT-ML
Rapidly exploring random trees with machine learning - learned sampling distributions, local reinforcement learning controller and learned supervised distance function for car-like mobile robots


<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li><a href="#interface">Interface</a></li>
        <li><a href="#configuring-an-experiment">Configuring an Experiment</a></li>
      </ul>
    </li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This project unites optimal rapidly exploring random trees (RRT*) with the following machine learning techniques:

* Learned distribution of robot configurations ([Itcher et. al., 2017](https://arxiv.org/abs/1709.05448))
* Reinforcement learning agent trained with DDPG ([Lillicrap et. al., 2015](https://arxiv.org/abs/1509.02971)) and MEGA ([Pitis et. al., 2020](https://arxiv.org/abs/2007.02832)) as a local controller
* Ordinal supervised learning ([Shi et. al. 2021](https://arxiv.org/abs/2111.08851)) of a distance metric induced by the agent

The experiments are conducted in PyBullet with a car-like mobile robot in a narrow passage type of scenario. 
The code allows the training and testing of each machine learning modules individually, but also in the context of the broader RRT* approach.

<!-- GETTING STARTED -->
## Getting Started

Follow the instructions below to install the package.

### Prerequisites

You need to install [PyTorch](https://pytorch.org/) and [PyBullet](https://pybullet.org/wordpress/).

#### Torch

The easiest way is to use conda.

If you have a CUDA-enabled GPU:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
```

If you want to use CPU only:
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

#### PyBullet

Installing PyBullet with pip requires build tools. I recommend using conda:

```
conda install -c conda-forge pybullet
```

### Installation

Install the package with pip:

```
pip install rrt-ml
```

<p align="right">(<a href="#rrt-ml">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

You can run experiments with different parameters for each module. Run from the command line or from any python file.

### Interface

Run experiments from the command line:

```
rrt-ml (--rl | --sl | --rrt) (--train | --test) [--hyper] [--config CONFIG]
```

1) The first option controls which algorithm will run:

* `--rl`: reinforcement learning agent as a local controller
* `--sl`: "sample learner" to learn sampling distributions for optimal motion planning with RRT*
* `--rrt`: optimal rapidly-exploring random tree 

2) The second option controls how should it run:

* `--train`: train a machine learning model (RL or SL) or grow the RRT* tree
* `--test`: generate various results (only after training)

3) The third option, `--hyper`, determines whether a search for hyperparameters should be made. If the second option is `--train`, then each possible config of the grid search is trained. If the second option is `--test`, then a bunch of results will be produced in order to compare different models (only after training).

4) The fourth option specify the name of the config. Below there are instructions to create a config file or directly run it, without using the command line.

### Configuring an Experiment

Create a python file anywhere. Create a `MasterConfig` object and set a name to it:

```
from rrt_ml.utilities.configs import MasterConfig


cfg = MasterConfig()

cfg.general.config_name_or_prefix = "MyExperimentConfigName"
```

Your IDE should auto-complete `cfg` and show all nested attributes. You can change various settings for all algorithms individually:

```
# Change reinforcement learning agent config
cfg.rl.general.gamma = 0.98
cfg.rl.actor.lr = 0.01
cfg.rl.critic.lr = 0.01

# Change sample learner (beta-cvae) config
cfg.sl.loss.beta = 5
cfg.sl.train.batch_size = 128

# Change RRT* config
cfg.rrt.general.seed = 1
cfg.rrt.sample.goal_prob = 0.05
cfg.rrt.names.rl = 'best'  # Use config name 'best' as controller for RRT
cfg.rrt.names.sl = 'best'  # Use config name 'best' as sample generator
```

You can also set a grid search over any setting that exists inside the `rl|sl|rrt` attributes above. Below an example on how to perform a grid search over the hyper-parameters for the RL agent:

```
cfg.hyperparams.rl.general.gamma = [0.95, 0.96, 0.97, 0.98, 0.99]
cfg.hyperparams.rl.actor.lr = [0.1, 0.01, 0.001]
cfg.hyperparams.rl.net.activ = ['ReLU', 'GeLU']
```

Now you can save the config to run later from the command line:

```
cfg.save()
```

Training an RL agent with this example config from the command line:

```
rrt-ml --rl --train --config=MyExperimentConfigName
```

Instead of saving and the running from the command line, you can simply run it from the python file you are editing, by adding this:

```
cfg.run(algorithm_to_run='rl', train_or_test='train', hyperparam_search_or_test=False)
```

The parameters of the `run` method are equal to the ones from the command line.

<p align="right">(<a href="#rrt-ml">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#rrt-ml">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [PyBullet](https://pybullet.org)
* [PyTorch](https://pytorch.org)
* [Modular RL](https://github.com/spitis/mrl)
* [Python Robotics](https://github.com/Lucifer2700/Python-Robotics)
* [Coral-PyTorch](https://github.com/Raschka-research-group/coral-pytorch)
* [TorchEnsemble](https://github.com/TorchEnsemble-Community/Ensemble-Pytorch)
* [Optuna](https://optuna.org)

<p align="right">(<a href="#rrt-ml">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 
