from typing import Protocol
import enum
import abc

import numpy as np
import numpy.typing as npt
from .variables import Var, VarName

class DifferentiableLossFunction(abc.ABC):

    @abc.abstractmethod
    def __call__(self, *args) -> float:
        """evaluate the differentiable loss function
        Parameters
        ----------
        *args

        Returns
        -------
        float
            scalar function value
        """
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self, *args) -> npt.NDArray:
        """gradient of the loss function with respect to a variable
        Parameters
        ----------
        *args

        Returns
        -------
        npt.NDArray
            gradient with the same shape as the variable

        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError

    def gradient_test(self,
                      *args,
                      eps: float = 1e-6,
                      inds: tuple = (0, ),
                      rtol: float = 1e-4,
                      atol: float = 1e-7) -> None:
        """check whether gradient matches its numerical approximation
        Parameters
        ----------
        eps : float, optional
            magnitude of pertubation, by default 1e-6
        inds : tuple, optional
            indicies along which to add / test pertubation, by default (0, )
        rtol : float, optional
            relative tolerance, by default 1e-4
        atol : float, optional
            absolute tolerance, by default 1e-7
        """
        pass


class DifferentiableLossScipy(abc.ABC):

    @abc.abstractmethod
    def __call__(self, current_estimate: npt.NDArray, *args) -> float:
        """ Evaluate differentiable loss function
        Parameters
        ----------
        current_estimate : npt.NDArray 1D
            current estimate for the variable var_name that is being optimized, ravelled
        var_dict : dict[str, npt.NDArray]
            all the variables
        var_name : str
            name of the variable being optimized
        Returns
        -------
        float
            the scalar loss function value
        """
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self, current_estimate: npt.NDArray, *args) -> npt.NDArray:
        """ Evaluate the gradient of the function with respect to a specified variable
        Parameters
        ----------
        current_estimate : npt.NDArray 1D
            current estimate for the variable var_name that is being optimized, ravelled
        var_dict : dict[str, npt.NDArray]
            all the variables
        var_name : str
            name of the variable being optimized
        Returns
        -------
        npt.NDArray 1D
            gradient with respect to variable "var_name", ravelled
        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError

    def gradient_test(self,
                      *args,
                      eps: float = 1e-6,
                      inds: tuple = (0, ),
                      rtol: float = 1e-4,
                      atol: float = 1e-7) -> None:
        """check whether gradient matches its numerical approximation
        Parameters
        ----------
        var_name : str
            name of variable for which gradient should be checked
        eps : float, optional
            magnitude of pertubation, by default 1e-6
        inds : tuple, optional
            indicies along which to add / test pertubation, by default (0, )
        rtol : float, optional
            relative tolerance, by default 1e-4
        atol : float, optional
            absolute tolerance, by default 1e-7
        """
        pass 
