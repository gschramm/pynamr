import enum
import numpy as np
from scipy.optimize import fmin_l_bfgs_b, fmin_cg

from .models import DualTESodiumAcqModel
from .models import TwoCompartmentBiExpDualTESodiumAcqModel
from .models import MonoExpDualTESodiumAcqModel

#-------------------------------------------------------------------


class CallingMode(enum.Enum):
    SINGLE = 'SINGLE'
    XFIRST = 'XFIRST'
    GAMFIRST = 'GAMFIRST'


class DataFidelityLoss:
    """ Data fidelity loss for mono exponential dual echo sodium forward model
        The model is linear in the first argument (the image x), but non linear
        in the second argument (Gam)
    """

    def __init__(self, model: DualTESodiumAcqModel, y: np.ndarray) -> None:
        self.model = model
        self.y = y

    def __call__(self, in1: np.ndarray, mode: CallingMode, *args) -> float:
        d = self.diff(in1, mode, *args)
        return 0.5 * (d**2).sum()

    def diff(self, in1: np.ndarray, mode: CallingMode, *args) -> np.ndarray:
        if isinstance(self.model, TwoCompartmentBiExpDualTESodiumAcqModel):
            x = in1
            diff = (self.model.forward(x.reshape(self.model.x_shape_real)) - self.y) * self.model.kmask

        elif isinstance(self.model, MonoExpDualTESodiumAcqModel):
            if len(args) == 0:
                raise TypeError('Data fidelity loss with MonoExpDualTESodiumAcqModel requires 3 input arguments.')

            if mode == CallingMode.XFIRST:
                x = in1
                gam = args[0]
            elif mode == CallingMode.GAMFIRST:
                x = args[0]
                gam = in1
            else:
                raise ValueError

            self.model.gam = gam
            diff = (self.model.forward(x.reshape(self.model.x_shape_real)) - self.y) * self.model.kmask

        else:
            raise NotImplementedError

        return diff

    def grad(self, in1: np.ndarray, mode: CallingMode, *args) -> np.ndarray:
        z = self.diff(in1, mode, *args)

        # reshaping of x/gam is needed since fmin_l_bfgs_b flattens all arrays
        if isinstance(self.model, TwoCompartmentBiExpDualTESodiumAcqModel):
            grad = self.model.adjoint(z).reshape(in1.shape)
        elif isinstance(self.model, MonoExpDualTESodiumAcqModel):
            if mode == CallingMode.XFIRST:
                # gradient with respect to x
                grad = self.model.adjoint(z).reshape(in1.shape)
            elif mode == CallingMode.GAMFIRST:
                # gradient with respect to gam
                self.model.gam = in1.reshape(self.model.image_shape)
                grad = self.model.grad_gam(z, args[0].reshape(self.model.x_shape_real)).reshape(in1.shape)
            else:
                raise ValueError
        else:
            raise NotImplementedError

        return grad


#-------------------------------------------------------------------


class TotalLoss:

    def __init__(self, datafidelityloss, penalty_x, penalty_gam, beta_x,
                 beta_gam):
        self.datafidelityloss = datafidelityloss
        self.penalty_x = penalty_x
        self.penalty_gam = penalty_gam

        self.beta_x = beta_x
        self.beta_gam = beta_gam

        if isinstance(self.datafidelityloss.model,
                      MonoExpDualTESodiumAcqModel):
            self.x_shape = self.datafidelityloss.model.image_shape + (2, )
        elif isinstance(self.datafidelityloss.model,
                        TwoCompartmentBiExpDualTESodiumAcqModel):
            self.x_shape = (2, ) + self.datafidelityloss.model.image_shape + (
                2, )

        self.gam_shape = self.datafidelityloss.model.image_shape

    def __call__(self, in1: np.ndarray, mode: CallingMode, *args) -> float:
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
                cost += self.beta_x * self.penalty_x.eval(
                    x.reshape(self.x_shape)[..., 0])
                cost += self.beta_x * self.penalty_x.eval(
                    x.reshape(self.x_shape)[..., 1])

            if self.beta_gam > 0:
                cost += self.beta_gam * self.penalty_gam.eval(gam)

        elif isinstance(self.datafidelityloss.model,
                        TwoCompartmentBiExpDualTESodiumAcqModel):
            x = in1

            if self.beta_x > 0:
                # reshaping of x is necessary since LBFGS will pass flattened arrays
                cost += self.beta_x * self.penalty_x.eval(
                    x.reshape(self.x_shape)[0, ..., 0])
                cost += self.beta_x * self.penalty_x.eval(
                    x.reshape(self.x_shape)[0, ..., 1])
                cost += self.beta_x * self.penalty_x.eval(
                    x.reshape(self.x_shape)[1, ..., 0])
                cost += self.beta_x * self.penalty_x.eval(
                    x.reshape(self.x_shape)[1, ..., 1])

        return cost

    def grad(self, in1: np.ndarray, mode: CallingMode, *args) -> np.ndarray:

        grad = self.datafidelityloss.grad(in1, mode, *args)

        # reshaping of x is necessary since LBFGS will pass flattened arrays
        if isinstance(self.datafidelityloss.model,
                      MonoExpDualTESodiumAcqModel):
            if mode == CallingMode.XFIRST:
                x = in1
                if self.beta_x > 0:
                    grad[..., 0] += self.beta_x * self.penalty_x.grad(
                        x.reshape(self.x_shape)[..., 0])
                    grad[..., 1] += self.beta_x * self.penalty_x.grad(
                        x.reshape(self.x_shape)[..., 1])
            elif mode == CallingMode.GAMFIRST:
                gam = in1
                if self.beta_gam > 0:
                    grad += self.beta_gam * self.penalty_gam.grad(gam)
            else:
                raise ValueError

        elif isinstance(self.datafidelityloss.model,
                        TwoCompartmentBiExpDualTESodiumAcqModel):
            x = in1
            if self.beta_x > 0:
                grad[0, ..., 0] += self.beta_x * self.penalty_x.grad(
                    x.reshape(self.x_shape)[0, ..., 0])
                grad[0, ..., 1] += self.beta_x * self.penalty_x.grad(
                    x.reshape(self.x_shape)[0, ..., 1])
                grad[1, ..., 0] += self.beta_x * self.penalty_x.grad(
                    x.reshape(self.x_shape)[1, ..., 0])
                grad[1, ..., 1] += self.beta_x * self.penalty_x.grad(
                    x.reshape(self.x_shape)[1, ..., 1])

        else:
            raise NotImplementedError

        return grad.reshape(in1.shape)
