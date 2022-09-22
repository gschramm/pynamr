
import enum
import numpy as np

#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------


class VarName(enum.Enum):
    IMAGE = 'Sodium image'
    GAMMA = 'Gamma'
    PARAM = 'Parameters of an image model'



class Var():
    """ Image space variable representing a parameter of the forward model """

    def __init__(self,
                 name: VarName,
                 shape: tuple,
                 nb_comp: int = 1,
                 complex_var: bool = True,
                 linearity: bool = True,
                 value: np.ndarray = None,
                 dtype: np.dtype = np.float64) -> None:


        # enum name
        self._name = name
        # shape = number of components + spatial dimensions + 2 for real and imaginary parts if complex
        self._shape = shape
        # real number type
        self._dtype = dtype
        # whether the forward model is linear with respect to this variable
        self._linearity = linearity
        # np.ndarray
        self._value = value
        # complex number variable
        self._complex_var = complex_var
        # Number of spatial components (e.g. multicompartmental voxel model)
        self._nb_comp = nb_comp



def putVarInFirstPlace(name: VarName, u: list[Var]) -> list[Var]:

    for ind, el in enumerate(u):
        if el._name==name:
            break

    if ind==len(u):
        raise IndexError

    if ind>0:
        temp = u[ind]
        u[ind] = u[0]
        u[0] = temp

    return u




