import matplotlib.pyplot as plt
import numpy as np

from rrt_ml.utilities.configs import *
from rrt_ml.utilities.paths import *


cfg = MasterConfig()

path = Paths().experiments / 'others' / 'alpha.png'

i = np.arange(1, 1000)
a, b, c = cfg.rrt.sample.learner_prob
sl_prob = a*np.exp(-(i - b)**2/(2*c**2))

fig, ax = plt.subplots()
ax.plot(i, sl_prob)
ax.grid(True)
ax.set_xlabel('Iteração')
ax.set_ylabel(r'$\alpha$')
ax.set_xlim([0, 500])

fig.savefig(path, dpi=600, bbox_inches='tight')
