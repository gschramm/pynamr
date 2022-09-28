from typing import Protocol
import enum
import abc

import numpy as np
import numpy.typing as npt
from .variables import Var, VarName


class DifferentiableFunction(abc.ABC):

    @abc.abstractmethod
    def __call__(self, arg1: npt.NDArray, *args) -> float:
        """evaluate differential function
        Parameters
        ----------
        arg1 : npt.NDArray
            first argument (needed for scipy optimize)
            it is the variable called "var_name"
        var_dict : dict[str, npt.NDArray]
            of all variables (first variable might be omitted)
        var_name : str
            name of first variable
        Returns
        -------
        float
            the function value
        """
        raise NotImplementedError

    @abc.abstractmethod
    def gradient(self, arg1: npt.NDArray, *args) -> npt.NDArray:
        """gradient of function with respect to a specific variable
        Parameters
        ----------
        arg1 : npt.NDArray
            first argument (needed for scipy optimize)
            it is the variable called "var_name"
        var_dict : dict[str, npt.NDArray]
            of all variables (first variable might be omitted)
        var_name : str
            name of first variable
        Returns
        -------
        npt.NDArray
            gradient with respect to variable "var_name"
        Raises
        ------
        NotImplementedError
            _description_
        """
        raise NotImplementedError

    def gradient_test(self,
                      var_name: VarName,
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

        var_dict = dict()

        for key, shape in self.input_shapes.items():
            var_dict[key] = np.random.rand(*shape)

        # calculate the loss with the original variables
        l1 = self.__call__(var_dict[var_name], var_dict, var_name)
        # calculate the gradient with the original variables
        g = self.gradient(var_dict[var_name], var_dict, var_name)

        for i in inds:
            # setup variables with small pertubation
            var_dict2 = deepcopy(var_dict)
            delta = np.zeros(self.input_shapes[var_name])
            delta.ravel()[i] = eps
            var_dict2[var_name] += delta

            # calculate loss with pertubed variables
            l2 = self.__call__(var_dict2[var_name], var_dict2, var_name)

            # approximate gradient
            g_approx = (l2 - l1) / eps

            assert (np.isclose(g[i], g_approx, rtol=rtol, atol=atol))


