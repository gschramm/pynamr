from typing import Protocol
import enum

import numpy as np

class CallingMode(enum.Enum):
    SINGLE = 'SINGLE'
    XFIRST = 'XFIRST'
    GAMFIRST = 'GAMFIRST'


class DifferentiableFunction(Protocol):
    def __call__(self, img: np.ndarray) -> float:
        ...
    def grad(self, img: np.ndarray) -> np.ndarray:
        ...