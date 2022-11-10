from typing import TypeAlias

import numpy as np


Vector: TypeAlias = list[float, ...] | tuple[float, ...] | np.ndarray
Vector2: TypeAlias = list[float, float] | tuple[float, float] | np.ndarray
Vector3: TypeAlias = list[float, float, float] | tuple[float, float, float] | np.ndarray
Vector4: TypeAlias = list[float, float, float, float] | tuple[float, float, float, float] | np.ndarray
Vector5: TypeAlias = list[float, float, float, float, float] | tuple[float, float, float, float, float] | np.ndarray
Vector6: TypeAlias = list[float, float, float, float, float, float] | tuple[float, float, float, float, float, float] \
                     | np.ndarray
