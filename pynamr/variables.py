
import enum
import numpy as np
from .utils import XpArray

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


class UnknownName(enum.Enum):
    IMAGE = 'sodium image'
    GAMMA = 'gamma'
    PARAM = 'parameters of the bicompartment biexp model'



class Unknown():

    def __init__(self,
                 name: UnknownName,
                 shape: tuple,
                 penaltyEntities: int = 1,
                 complexNb: bool = True,
                 linearity: bool = True,
                 value: XpArray = None,
                 dtype: np.dtype = np.float64) -> None:


        # shape in complex form
        self._name = name
        self._shape = shape
        self._dtype = dtype
        self._linearity = linearity
        self._value = value
        self._complex = complexNb
        self._penaltyEntities = penaltyEntities


