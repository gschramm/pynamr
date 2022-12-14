import numpy as np
import numpy.typing as npt
from copy import deepcopy

from .models import DualTESodiumAcqModel
from .models import TwoCompartmentBiExpDualTESodiumAcqModel
from .models import MonoExpDualTESodiumAcqModel
from .models import Var, VarName

from .protocols import DifferentiableLossFunction, DifferentiableLossScipy

from IPython.core.debugger import set_trace

class DataFidelityLoss(DifferentiableLossFunction):
    """ Data fidelity loss for sodium MRI forward models.
    """

    def __init__(self, model: DualTESodiumAcqModel, data: npt.NDArray) -> None:
        """data fidelity loss

        Parameters
        ----------
        model : DualTESodiumAcqModel
            acquisition model for dual TE sodium acquisition
        data : np.ndarray
            acquired data
        """        
        self._data = data
        self._model = model

    def __call__(self, var_dict: dict[VarName, Var]) -> float:
        """calculate data fidelity loss

        Parameters
        ----------

        var_dict : dict[VarName,Var]
            dictionary of image space variables that represent known or unknown parameters of the forward model
            (i.e. images, image model parameters, T2* maps, Gamma)

        Returns
        -------
        float
            the scalar loss value
        """        
        return 0.5 * ((self._model.forward(var_dict) - self._data)**2).sum()

    def gradient(self, var_dict: dict[VarName, Var], var_name: VarName) -> npt.NDArray:
        """calculate the gradient of the data fidelity loss with respect to the variable var_name

        Parameters
        ----------
        var_dict : dict[VarName,Var]
            all the variables that represent known or unknown parameters of the forward model
            (i.e. images, image model parameters, T2* maps, Gamma)

        Returns
        -------
        np.ndarray
            the gradient, shaped as the variable
        """
        outer = (self._model.forward(var_dict) - self._data)
        return self._model.gradient(outer, var_dict, var_name)

    def gradient_test(self,
                      var_dict: dict[VarName,Var],
                      var_name,
                      eps: float = 1e-5,
                      inds: tuple = (0, ),
                      rtol: float = 1e-4,
                      atol: float = 1e-7) -> None:
        """check whether gradient matches its numerical approximation
        Parameters
        ----------
        var_dict : dict[str, npt.NDArray]
            all the variables
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

        for key, var in var_dict.items():
            var.value = np.random.rand(*var.shape)

        # calculate the loss with the original variables
        l1 = self.__call__(var_dict)
        # calculate the gradient with the original variables
        g = self.gradient(var_dict, var_name)

        # compute indices for the unknown variable/loss gradient for all the relevant tests:
        # specified spatial indices, real/imag part if complex number, all the components
        indices = []
        for i in inds:
            sl = (i,) * var_dict[var_name].nb_spatial
            if var_dict[var_name].nb_comp>1:
                for c in range(var_dict[var_name].nb_comp):
                    slc = (c,) + sl
                    if var_dict[var_name].complex_var:
                        for j in range(2):
                            indices.append(slc + (j,))
                    else:
                        indices.append(slc)
            elif var_dict[var_name].complex_var:
                for j in range(2):
                    indices.append(sl + (j,))
            else:
                indices.append(sl)
 
        # compare approximate and actual gradient
        for item in indices:
            print(item)
            # setup variables with small pertubation
            var_dict2 = deepcopy(var_dict)
            delta = np.zeros(var_dict[var_name].shape)

            delta[item] = eps
            var_dict2[var_name].value += delta

            # calculate loss with pertubed variables
            l2 = self.__call__(var_dict2)

            # approximate gradient
            g_approx = (l2 - l1) / eps

            assert (np.isclose(g[item], g_approx, rtol=rtol, atol=atol))


class TotalLoss(DifferentiableLossScipy):
    """Total loss function to be optimized, consisting of data fidelity and penalties
       Complies with the interface required for using scipy.optimize.minimize
    """
    def __init__(self,
                 datafidelityloss: DataFidelityLoss,
                 penalties: dict[VarName, DifferentiableLossFunction],
                 betas: dict[VarName, float] ):
        """
        Parameters
        ----------
        datafidelityloss : DataFidelityLoss
            object to calculate data fidelity loss and gradient
        penalties : dictionary containing functions for calculating penalties for relevant variables
        betas : dictionary containing penalty weights for relevant variables
        """                 

        self.datafidelityloss = datafidelityloss
        self.penalties = penalties
        self.betas = betas


    def __call__(self, current_estimate: np.ndarray, var_dict: dict[VarName,Var], var_name: VarName) -> float:
        """ Evaluate differentiable loss function
        Parameters
        ----------
        current_estimate : npt.NDArray
            current estimate for the variable var_name that is being optimized
        var_dict : dict[str, npt.NDArray]
            all the variables
        var_name : str
            name of the variable being optimized
        Returns
        -------
        float
            the scalar loss function value
        """

        # update the variable list with the current value
        var_dict[var_name].value = current_estimate.reshape(var_dict[var_name].shape)

        # data fidelity loss
        cost = self.datafidelityloss(var_dict)

        # add all the penalties
        for name, var in var_dict.items():
            if (name in self.penalties) and (self.betas[name]>0):
                if var.complex_var:
                    for j in range(2):
                        if var.nb_comp>1:
                            for k in range(var.nb_comp):
                                cost += self.betas[name] * self.penalties[name](var.value[k,...,j])
                        else:
                            cost += self.betas[name] * self.penalties[name](var.value[...,j])
                else:
                    if var.nb_comp>1:
                        for k in range(var.nb_comp):
                             cost += self.betas[name] * self.penalties[name](var.value[k])
                    else:
                        cost += self.betas[name] * self.penalties[name](var.value)

        return cost

    def gradient(self, current_estimate: np.ndarray, var_dict: dict[VarName,Var], var_name: VarName) -> np.ndarray:
        """ Evaluate the gradient of the function with respect to the specified variable
        Parameters
        ----------I
        current_estimate : npt.NDArray
            current estimate for the variable var_name that is being optimized
        var_dict : dict[str, npt.NDArray]
            all the variables
        var_name : str
            name of the variable being optimized

        Returns
        -------
        npt.NDArray 1D
            gradient with respect to variable "var_name", ravelled
        """

        # update the variable list with the current value
        var_dict[var_name].value = current_estimate.reshape(var_dict[var_name].shape)

        # data fidelity loss gradient
        grad = self.datafidelityloss.gradient(var_dict, var_name)

        # add penalty gradient for the first variable
        var = var_dict[var_name]
        if (var_name in self.penalties) and (self.betas[var_name]>0):
            if var.complex_var:
                for j in range(2):
                    if var.nb_comp>1:
                        for k in range(var.nb_comp):
                            grad[k,...,j] += self.betas[var_name] * self.penalties[var_name].gradient(var.value[k,...,j])
                    else:
                        grad[...,j] += self.betas[var_name] * self.penalties[var_name].gradient(var.value[...,j])
            else:
                if var.nb_comp>1:
                    for k in range(var.nb_comp):
                        grad[k] += self.betas[var_name] * self.penalties[var_name].gradient(var.value[k])
                else:
                    grad += self.betas[var_name] * self.penalties[var_name].gradient(var.value)

        # flatten the array for scipy.optimize
        return grad.ravel()


    def gradient_test(self,
                      var_dict: dict[VarName,Var],
                      var_name,
                      eps: float = 1e-5,
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

        for key, var in var_dict.items():
            var.value = np.random.rand(*var.shape)

        # calculate the loss with the original variables
        l1 = self.__call__(var_dict[var_name].value.ravel(), var_dict, var_name)
        # calculate the gradient with the original variables
        g = self.gradient(var_dict[var_name].value.ravel(), var_dict, var_name)
        # have to reshape it as the total loss gradient is ravelled because it serves
        # as interface to scipy
        g = g.reshape(var_dict[var_name].shape)

        # compute indices for the unknown variable/loss gradient for all the relevant tests:
        # specified spatial indices, real/imag part if complex number, all the components
        indices = []
        for i in inds:
            sl = (i,) * var_dict[var_name].nb_spatial
            if var_dict[var_name].nb_comp>1:
                for c in range(var_dict[var_name].nb_comp):
                    slc = (c,) + sl
                    if var_dict[var_name].complex_var:
                        for j in range(2):
                            indices.append(slc + (j,))
                    else:
                        indices.append(slc)
            elif var_dict[var_name].complex_var:
                for j in range(2):
                    indices.append(sl + (j,))
            else:
                indices.append(sl)

        # compare approximate and actual gradient
        for item in indices:
            # setup variables with small pertubation
            var_dict2 = deepcopy(var_dict)
            delta = np.zeros(var_dict[var_name].shape)
            delta[item] = eps
            var_dict2[var_name].value += delta

            # calculate loss with pertubed variables
            l2 = self.__call__(var_dict2[var_name].value.ravel(), var_dict2, var_name)
            # approximate gradient
            g_approx = (l2 - l1) / eps

            assert (np.isclose(g[item], g_approx, rtol=rtol, atol=atol))

