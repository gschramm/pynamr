import numpy as np

from .models import DualTESodiumAcqModel
from .models import TwoCompartmentBiExpDualTESodiumAcqModel
from .models import MonoExpDualTESodiumAcqModel

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

    def __call__(self, in1: np.ndarray, mode: CallingMode, *args) -> float:
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
        d = self.diff(in1, mode, *args)
        return 0.5 * (d**2).sum()

    def diff(self, in1: np.ndarray, mode: CallingMode, *args) -> np.ndarray:
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
        if isinstance(self.model, TwoCompartmentBiExpDualTESodiumAcqModel):
            x = in1
            diff = (self.model.forward(x.reshape(self.model.x_shape_real)) -
                    self.y) * self.model.kmask

        elif isinstance(self.model, MonoExpDualTESodiumAcqModel):
            if len(args) == 0:
                raise TypeError(
                    'Data fidelity loss with MonoExpDualTESodiumAcqModel requires 3 input arguments.'
                )

            if mode == CallingMode.XFIRST:
                x = in1
                gam = args[0]
            elif mode == CallingMode.GAMFIRST:
                x = args[0]
                gam = in1
            else:
                raise ValueError

            self.model.gam = gam.reshape(self.model.image_shape)
            diff = (self.model.forward(x.reshape(self.model.x_shape_real)) -
                    self.y) * self.model.kmask

        else:
            raise NotImplementedError

        return diff

    def grad(self, in1: np.ndarray, mode: CallingMode, *args) -> np.ndarray:
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
        z = self.diff(in1, mode, *args)

        # reshaping of x/gam is needed since fmin_l_bfgs_b flattens all arrays
        if isinstance(self.model, TwoCompartmentBiExpDualTESodiumAcqModel):
            grad = self.model.adjoint(z).reshape(in1.shape)
        elif isinstance(self.model, MonoExpDualTESodiumAcqModel):
            if len(args) == 0:
                raise TypeError(
                    'Data fidelity loss gradient with MonoExpDualTESodiumAcqModel requires 3 input arguments.'
                )

            if mode == CallingMode.XFIRST:
                # gradient with respect to x
                grad = self.model.adjoint(z).reshape(in1.shape)
            elif mode == CallingMode.GAMFIRST:
                # gradient with respect to gam
                self.model.gam = in1.reshape(self.model.image_shape)
                grad = self.model.grad_gam(
                    z, args[0].reshape(self.model.x_shape_real)).reshape(
                        in1.shape)
            else:
                raise ValueError
        else:
            raise NotImplementedError

        return grad


class TotalLoss:

    def __init__(self, 
                 datafidelityloss: DataFidelityLoss, 
                 penalty_x : DifferentiableFunction, 
                 beta_x: float, 
                 penalty_gam: DifferentiableFunction | None = None,
                 beta_gam: float = 0) -> None:
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
        self.penalty_x = penalty_x
        self.penalty_gam = penalty_gam

        self.beta_x = beta_x
        self.beta_gam = beta_gam

        self.x_shape = self.datafidelityloss.model.x_shape_real

        self.gam_shape = self.datafidelityloss.model.image_shape

    def __call__(self, in1: np.ndarray, mode: CallingMode, *args) -> float:
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
        cost = self.datafidelityloss(in1, mode, *args)

        if isinstance(self.datafidelityloss.model,
                      MonoExpDualTESodiumAcqModel):
            if mode == CallingMode.XFIRST:
                x = in1
                gam = args[0]
            elif mode == CallingMode.GAMFIRST:
                x = args[0]
                gam = in1
            else:
                raise ValueError

            if self.beta_x > 0:
                # reshaping of x is necessary since LBFGS will pass flattened arrays
                for ch in range(self.x_shape[0]):
                    for j in range(2):
                        cost += self.beta_x * self.penalty_x(
                            x.reshape(self.x_shape)[ch, ..., j])

            if (self.beta_gam > 0) and (self.penalty_gam is not None):
                cost += self.beta_gam * self.penalty_gam(gam)

        elif isinstance(self.datafidelityloss.model,
                        TwoCompartmentBiExpDualTESodiumAcqModel):
            x = in1

            if self.beta_x > 0:
                # reshaping of x is necessary since LBFGS will pass flattened arrays
                for ch in range(self.x_shape[0]):
                    for j in range(2):
                        cost += self.beta_x * self.penalty_x(
                            x.reshape(self.x_shape)[ch, ..., j])

        return cost

    def grad(self, in1: np.ndarray, mode: CallingMode, *args) -> np.ndarray:
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

        grad = self.datafidelityloss.grad(in1, mode, *args)

        # calculate the gradient with respect to the "smoothing" penalties
        if isinstance(self.datafidelityloss.model,
                      MonoExpDualTESodiumAcqModel):
            if mode == CallingMode.XFIRST:
                x = in1
                # reshaping of data fidelity gradient is necessary since fmin_l_bfgs
                # passes flattened arrays
                grad = grad.reshape(self.datafidelityloss.model.x_shape_real) 

                if self.beta_x > 0:
                    for ch in range(self.x_shape[0]):
                        for j in range(2):
                            grad[ch, ...,
                                 j] += self.beta_x * self.penalty_x.grad(
                                     x.reshape(self.x_shape)[ch, ..., j])

                grad = grad.reshape(in1.shape)
            elif mode == CallingMode.GAMFIRST:
                gam = in1
                if (self.beta_gam > 0) and (self.penalty_gam is not None):
                    grad += self.beta_gam * self.penalty_gam.grad(gam)
                
                # unclear why the ravel is needed, but without fmin_l_bfgs is not happy
                if grad.ndim == 1:
                    grad = grad.ravel()
            else:
                raise ValueError

        elif isinstance(self.datafidelityloss.model,
                        TwoCompartmentBiExpDualTESodiumAcqModel):
            x = in1
            if self.beta_x > 0:
                grad = grad.reshape(self.datafidelityloss.model.x_shape_real) 

                for ch in range(self.x_shape[0]):
                    for j in range(2):
                        grad[ch, ..., j] += self.beta_x * self.penalty_x.grad(
                            x.reshape(self.x_shape)[ch, ..., j])
                
                grad = grad.reshape(in1.shape)
        else:
            raise NotImplementedError
        
        return grad
