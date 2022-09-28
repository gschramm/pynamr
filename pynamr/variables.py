
import enum
import numpy as np
from dataclasses import dataclass
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------

class VarName(enum.Enum):
    IMAGE = 'Sodium image'
    GAMMA = 'Gamma'
    PARAM = 'Parameters of an image model'



@dataclass
class Var():
    """ Image space variable representing a parameter of the forward model """

    # shape = number of components + spatial dimensions + 2 for real and imaginary parts if complex
    shape: tuple
    # Number of spatial components (e.g. multicompartmental voxel model)
    nb_comp: int = 1
    # complex number variable: the last dimension is the real and imaginary component
    complex_var: bool = True
    # whether the forward model is linear with respect to this variable
    linearity: bool = True
    # np.ndarray
    value: np.ndarray = None
    # real number type
    dtype: np.dtype = np.float64


