import numpy as np
import numpy.typing as npt
from copy import deepcopy

from .models import DualTESodiumAcqModel
from .models import TwoCompartmentBiExpDualTESodiumAcqModel
from .models import MonoExpDualTESodiumAcqModel
from .models import Var, VarName

from .protocols import DifferentiableFunction


class DataFidelityLoss(DifferentiableFunction):
    """ Data fidelity loss for mono exponential and bi exponential dual echo 
        sodium forward model.
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

    def __call__(self, arg1: npt.NDArray, var_dict: dict[VarName, Var], var_name: VarName) -> float:
        """calculate data fidelity loss

        Parameters
        ----------

        u : dict[VarName,Var]
            the list of image space variables that represent known or unknown parameters of the forward model
            (i.e. images, image model parameters, T2* maps, Gamma)

        Returns
        -------
        float
            the loss value
        """        
        var_dict[var_name].value = np.reshape(arg1, var_dict[var_name].shape)
        return 0.5 * ((self._model.forward(var_dict) - self._data)**2).sum()

    def gradient(self, arg1: npt.NDArray, var_dict: dict[VarName, Var],
                 var_name: VarName) -> npt.NDArray:
        """calculate the gradient of the data fidelity loss with respect to the first variable in the list

        Parameters
        ----------
        u : dict[VarName,Var]
            the list of image space variables that represent known or unknown parameters of the forward model
            (i.e. images, image model parameters, T2* maps, Gamma)

        Returns
        -------
        np.ndarray
            the gradient

        Note
        ----
        """

        var_dict[var_name].value = np.reshape(arg1, var_dict[var_name].shape)
        outer = (self._model.forward(var_dict) - self._data)
        return self._model.gradient(outer, var_dict, var_name)

    def gradient_test(self,
                      var_name: VarName,
                      eps: float = 1e-6,
                      inds: tuple = (0, ),
                      rtol: float = 1e-4,
                      atol: float = 1e-7) -> None:
        pass


class TotalLoss(DifferentiableFunction):
    """Total loss function to be optimized consisting of data fidelity and priors
      Complies with the interface required for using scipy.optimize
    """
    def __init__(self,
                 datafidelityloss: DataFidelityLoss,
                 penalties: dict[VarName, DifferentiableFunction],
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


    def __call__(self, in1: np.ndarray, var_dict: dict[VarName,Var], var_name: VarName) -> float:
        """calculate total loss

        Parameters
        ----------
        in1 : np.ndarray
            current value of the variable being optimized
        *args : additional input arguments, here dict[VarName,Var]

        Returns
        -------
        float
            the loss value

        Note
        -------
        Interface for scipy.optimize
        """

        # deep copy because scipy.optimize requires fixed additional arguments
        # update the variable list with the current value
        var_dict[var_name].value = in1.reshape(var_dict[var_name].shape)

        # data fidelity loss
        cost = self.datafidelityloss(in1, var_dict, var_name)

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

    def gradient(self, in1: np.ndarray, var_dict: dict[VarName,Var], var_name: VarName) -> np.ndarray:
        """calculate the gradient of the total loss with respect to the first variable in the list

        Parameters
        ----------
        in1 : np.ndarray
            current value of the variable being optimized
        *args : additional input arguments, here dict[VarName,Var]


        Returns
        -------
        np.ndarray
            the gradient

        Note
        ----
        Interface for scipy.optimize
        """

        # deep copy because scipy.optimize requires fixed additional arguments
        # update the variable list with the current value
        var_dict[var_name].value = in1.reshape(var_dict[var_name].shape)

        # data fidelity loss gradient
        grad = self.datafidelityloss.gradient(in1, var_dict, var_name)

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
