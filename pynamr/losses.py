import numpy as np

from .models import DualTESodiumAcqModel
from .models import TwoCompartmentBiExpDualTESodiumAcqModel
from .models import MonoExpDualTESodiumAcqModel
from .models import Unknown

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

    def __call__(self, u: list[Unknown]) -> float:
        """calculate data fidelity loss

        Parameters
        ----------

        u : list[Unknown]
            the list of image space variables that represent known or unknown parameters of the forward model
            (i.e. images, image model parameters, T2* maps)

        Returns
        -------
        float
            the loss value
        """        
        d = self.diff(u)
        return 0.5 * (d**2).sum()

    def diff(self, u: list[Unknown]) -> np.ndarray:
        """calculate the bin-wise difference between the data and the expectation

        Parameters
        ----------
        u : list[Unknown]
            the list of image space variables that represent known or unknown parameters of the forward model
            (i.e. images, image model parameters, T2* maps)

        Returns
        -------
        np.ndarray
            bin-wise difference between data and expected data
        """

        diff = (self.model.forward(u) -
                    self.y) * self.model.kmask

        return diff

    def grad(self, u: list[Unknown]) -> np.ndarray:
        """calculate the gradient of the data fidelity loss with respect to the first variable in the list

        Parameters
        ----------
        u : list[Unknown]
            the list of image space variables that represent known or unknown parameters of the forward model
            (i.e. images, image model parameters, T2* maps)

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
                 penalties: dict[str,DifferentiableFunction],
                 betas: dict[str,float] ):
        """
        Parameters
        ----------
        datafidelityloss : DataFidelityLoss
            object to calculate data fidelity loss and gradient
        penalties : list[DifferentiableFunction}
            functions for calculating penalties
        betas : float, optional
            penalty weights
        """                 

        self.datafidelityloss = datafidelityloss
        self.penalties = penalties
        self.betas = betas


    def __call__(self, in1: np.ndarray, *args: list[Unknown]) -> float:
        """calculate total loss

        Parameters
        ----------
        in1 : np.ndarray
            current value of the variable being optimized
        *args : additional input arguments, here list[Unknown]

        Returns
        -------
        float
            the loss value

        Note
        -------
        Interface for scipy.optimize
        """

        u = args
        # update the variable list with the current value for consistency and ease of use
        u[0]._value = in1.reshape(u[0]._shape)

        cost = self.datafidelityloss(u)

        for el in u:
            if (self.betas[el._name]>0) and (self.penalties[el._name] is not None):
                if el._complex_var:
                    for j in range(2):
                        if el.nb_comp>1:
                            for k in range(el.nb_comp):
                                cost += self.betas[el._name] * self.penalties[el._name](el._value[k,...,j])
                            else:
                                cost += self.betas[el._name] * self.penalties[el._name](el._value[...,j])
                else:
                    if el.nb_comp>1:
                        for k in range(el.nb_comp):
                             cost += self.betas[el._name] * self.penalties[el._name](el._value[k])
                    else:
                        cost += self.betas[el._name] * self.penalties[el._name](el._value)

        return cost

    def grad(self, in1: np.ndarray, *args: list[Unknown]) -> np.ndarray:
        """calculate the gradient of the total loss with respect to the first variable in the list

        Parameters
        ----------
        in1 : np.ndarray
            current value of the variable being optimized
        *args : additional input arguments, here list[Unknown]


        Returns
        -------
        np.ndarray
            the gradient

        Note
        ----
        Interface for scipy.optimize
        """

        u = args
        # update the variable list with the current value for consistency and ease of use
        u[0]._value = in1.reshape(u[0]._shape)

        # data fidelity loss gradient
        grad = self.datafidelityloss.grad(u)

        # add penalty gradient for the first variable
        el = u[0]
        if (self.betas[el._name]>0) and (self.penalties[el._name] is not None):
            if el._complex_var:
                for j in range(2):
                    if el.nb_comp>1:
                        for k in range(el.nb_comp):
                            grad[k,...,j] += self.betas[el._name] * self.penalties[el._name].grad(el._value[k,...,j])
                        else:
                            grad[...,j] += self.betas[el._name] * self.penalties[el._name].grad(el._value[...,j])
            else:
                if el.nb_comp>1:
                    for k in range(el.nb_comp):
                        grad[k] += self.betas[el._name] * self.penalties[el._name].grad(el._value[k])
                else:
                    grad += self.betas[el._name] * self.penalties[el._name].grad(el._value)

        # flatten the array for scipy.optimize
        return grad.ravel()
