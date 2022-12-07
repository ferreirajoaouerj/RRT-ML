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
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This project unites optimal rapidly exploring random trees (RRT*) with the following machine learning techniques:

* Learned distribution of samples (robot configurations) as proposed by [@cite itcher] 
* Reinforcement learning agent trained with MEGA [@cite spitis] as a local controller
* Supervised learning of a distance metric induced by the agent

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

Run experiments from the command line:

```
rrt-ml -h
```

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
