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
        in1 : np.ndarray
            either the image(s) x for Mono of BiExp models or the decay image gamma
            for the MonoExp model
        mode : CallingMode
            that signals whether the image x or the decay image gamma was passed as
            first argument
        *args : additional input arguments
            For the MonoExp model the "second" image has to be passed as *args[0]

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
        in1 : np.ndarray
            either the image(s) x for Mono of BiExp models or the decay image gamma
            for the MonoExp model
        mode : CallingMode
            that signals whether the image x or the decay image gamma was passed as
            first argument
        *args : additional input arguments
            For the MonoExp model the "second" image has to be passed as *args[0]

        Returns
        -------
        np.ndarray
            bin-wise difference between data and expected data
        """

        diff = (self.model.forward(u) -
                    self.y) * self.model.kmask

        return diff

    def grad(self, u: list[Unknown]) -> np.ndarray:
        """calculate the gradient of the data fidelity loss

        Parameters
        ----------
        in1 : np.ndarray
            either the image(s) x for Mono of BiExp models or the decay image gamma
            for the MonoExp model
        mode : CallingMode
            that signals whether the image x or the decay image gamma was passed as
            first argument
        *args : additional input arguments
            For the MonoExp model the "second" image has to be passed as *args[0]

        Returns
        -------
        np.ndarray
            the gradient

        Note
        ----
        The gradient is calculate with respect to the first input argument.
        In that way this function can be used to calculate the gradient with
        respect to x and gamma for the MonoExp signal model.
        """

        u[0]._value = in1.reshape(u[0]._shape)

        z = self.diff(u)

        # reshaping of x/gam is needed since fmin_l_bfgs_b flattens all arrays
        if u[0]._linearity:
            grad = self.model.adjoint(z,u)
        else:
            grad = self.model.grad(z,u)

        return grad


class TotalLoss:

    def __init__(self, 
                 datafidelityloss: DataFidelityLoss, 
                 penalties: dict[str,DifferentiableFunction],
                 betas: dict[str,float] ):


        """Total loss function to be optimized consisting of data fidelity and priors

        Parameters
        ----------
        datafidelityloss : DataFidelityLoss
            object to calculate data fidelity loss and gradient
        penalty_x : DifferentiableFunction
            object to calculate penalty on x and gradient
        beta_x : float, optional
            weight for penalty on x
        penalty_gam : DifferentiableFunction | None, optional
            object to calculate penalty on decay image gamma
        beta_gam : float | None, optional
            weight for penalty on gamma
        """                 

        self.datafidelityloss = datafidelityloss
        self.penalties = penalties
        self.betas = betas


    def __call__(self, in1: np.ndarray, *args) -> float:
        """calculate total loss

        Parameters
        ----------
        in1 : np.ndarray
            either the image(s) x for Mono of BiExp models or the decay image gamma
            for the MonoExp model
        mode : CallingMode
            that signals whether the image x or the decay image gamma was passed as
            first argument
        *args : additional input arguments
            For the MonoExp model the "second" image has to be passed as *args[0]

        Returns
        -------
        float
            the loss value
        """

        u = args

        u[0]._value = in1.reshape(u[0]._shape)

        cost = self.datafidelityloss(u)

        for el in u:
            if (self.betas[el._name]>0) and (self.penalties[el._name] is not None):
                if el._complex:
                    for j in range(2):
                        if el._penaltyEntities>1:
                            for k in range(el._penaltyEntities):
                                cost += self.betas[el._name] * self.penalties[el._name](el._value[k,...,j])
                            else:
                                cost += self.betas[el._name] * self.penalties[el._name](el._value[...,j])
                else:
                    if el._penaltyEntities>1:
                            for k in range(el._penaltyEntities):
                                 cost += self.betas[el._name] * self.penalties[el._name](el._value[k])
                    else:
                        cost += self.betas[el._name] * self.penalties[el._name](el._value)

        return cost

    def grad(self, in1: np.ndarray, *args) -> np.ndarray:
        """calculate the gradient of the total loss

        Parameters
        ----------
        in1 : np.ndarray
            either the image(s) x for Mono of BiExp models or the decay image gamma
            for the MonoExp model
        mode : CallingMode
            that signals whether the image x or the decay image gamma was passed as
            first argument
        *args : additional input arguments
            For the MonoExp model the "second" image has to be passed as *args[0]

        Returns
        -------
        np.ndarray
            the gradient

        Note
        ----
        The gradient is calculate with respect to the first input argument.
        In that way this function can be used to calculate the gradient with
        respect to x and gamma for the MonoExp signal model.
        """

        u = args
        u[0]._value = reshape(u[0]._shape)

        grad = self.datafidelityloss.grad(u)

        for el in u:
            if (self.betas[el._name]>0) and (self.penalties[el._name] is not None):
                if el._complex:
                    for j in range(2):
                        if el._penaltyEntities>1:
                            for k in range(el._penaltyEntities):
                                grad += self.betas[el._name] * self.penalties[el._name].grad(el._value[k,...,j])
                            else:
                                grad += self.betas[el._name] * self.penalties[el._name].grad(el._value[...,j])

        return grad.ravel()
