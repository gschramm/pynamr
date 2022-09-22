import numpy as np
from copy import deepcopy

from .models import DualTESodiumAcqModel
from .models import TwoCompartmentBiExpDualTESodiumAcqModel
from .models import MonoExpDualTESodiumAcqModel
from .models import Var, VarName

from .protocols import DifferentiableFunction, CallingMode

class DataFidelityLoss:
    """ Data fidelity loss for mono exponential and bi exponential dual echo 
        sodium forward model.
    """

    def __init__(self, model: DualTESodiumAcqModel, y: np.ndarray) -> None:
        """data fidelity loss

        Parameters
        ----------
        model : DualTESodiumAcqModel
            acquisition model for dual TE sodium acquisition
        y : np.ndarray
            acquired data
        """        
        self.model = model
        self.y = y

    def __call__(self, u: list[Var]) -> float:
        """calculate data fidelity loss

        Parameters
        ----------

        u : list[Var]
            the list of image space variables that represent known or unknown parameters of the forward model
            (i.e. images, image model parameters, T2* maps, Gamma)

        Returns
        -------
        float
            the loss value
        """        
        d = self.diff(u)
        return 0.5 * (d**2).sum()

    def diff(self, u: list[Var]) -> np.ndarray:
        """calculate the bin-wise difference between the data and the expectation

        Parameters
        ----------
        u : list[Var]
            the list of image space variables that represent known or unknown parameters of the forward model
            (i.e. images, image model parameters, T2* maps, Gamma)

        Returns
        -------
        np.ndarray
            bin-wise difference between data and expected data
        """

        diff = (self.model.forward(u) -
                    self.y) * self.model.kmask

        return diff

    def grad(self, u: list[Var]) -> np.ndarray:
        """calculate the gradient of the data fidelity loss with respect to the first variable in the list

        Parameters
        ----------
        u : list[Var]
            the list of image space variables that represent known or unknown parameters of the forward model
            (i.e. images, image model parameters, T2* maps, Gamma)

        Returns
        -------
        np.ndarray
            the gradient

        Note
        ----
        """

        z = self.diff(u)

        # if model linear with respect to the variable call simply the adjoint, otherwise call more general gradient computation
        if u[0]._linearity:
            grad = self.model.adjoint(z,u)
        else:
            grad = self.model.grad(z,u)

        return grad


class TotalLoss:
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


    def __call__(self, in1: np.ndarray, *args: list[Var]) -> float:
        """calculate total loss

        Parameters
        ----------
        in1 : np.ndarray
            current value of the variable being optimized
        *args : additional input arguments, here list[Var]

        Returns
        -------
        float
            the loss value

        Note
        -------
        Interface for scipy.optimize
        """

        # deep copy because scipy.optimize requires fixed additional arguments
        u = deepcopy(args)
        # update the variable list with the current value
        u[0]._value = in1.reshape(u[0]._shape)

        # data fidelity loss
        cost = self.datafidelityloss(u)

        # add all the penalties
        for el in u:
            if (el._name in self.penalties) and (self.betas[el._name]>0):
                if el._complex_var:
                    for j in range(2):
                        if el._nb_comp>1:
                            for k in range(el._nb_comp):
                                cost += self.betas[el._name] * self.penalties[el._name](el._value[k,...,j])
                            else:
                                cost += self.betas[el._name] * self.penalties[el._name](el._value[...,j])
                else:
                    if el._nb_comp>1:
                        for k in range(el._nb_comp):
                             cost += self.betas[el._name] * self.penalties[el._name](el._value[k])
                    else:
                        cost += self.betas[el._name] * self.penalties[el._name](el._value)

        return cost

    def grad(self, in1: np.ndarray, *args: list[Var]) -> np.ndarray:
        """calculate the gradient of the total loss with respect to the first variable in the list

        Parameters
        ----------
        in1 : np.ndarray
            current value of the variable being optimized
        *args : additional input arguments, here list[Var]


        Returns
        -------
        np.ndarray
            the gradient

        Note
        ----
        Interface for scipy.optimize
        """

        # deep copy because scipy.optimize requires fixed additional arguments
        u = deepcopy(args)
        # update the variable list with the current value
        u[0]._value = in1.reshape(u[0]._shape)

        # data fidelity loss gradient
        grad = self.datafidelityloss.grad(u)

        # add penalty gradient for the first variable
        el = u[0]
        if (el._name in self.penalties) and (self.betas[el._name]>0):
            if el._complex_var:
                for j in range(2):
                    if el._nb_comp>1:
                        for k in range(el._nb_comp):
                            grad[k,...,j] += self.betas[el._name] * self.penalties[el._name].grad(el._value[k,...,j])
                        else:
                            grad[...,j] += self.betas[el._name] * self.penalties[el._name].grad(el._value[...,j])
            else:
                if el._nb_comp>1:
                    for k in range(el._nb_comp):
                        grad[k] += self.betas[el._name] * self.penalties[el._name].grad(el._value[k])
                else:
                    grad += self.betas[el._name] * self.penalties[el._name].grad(el._value)

        # flatten the array for scipy.optimize
        return grad.ravel()
