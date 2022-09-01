#TODO: - replace interface to loss with __call__(x, method = 'x_first/gam_first')

import numpy as np
from scipy.optimize import fmin_l_bfgs_b, fmin_cg

#-------------------------------------------------------------------


class DataFidelityLoss:
    """ Data fidelity loss for mono exponential dual echo sodium forward model
        The model is linear in the first argument (the image x), but non linear
        in the second argument (Gam)
    """

    def __init__(self, model, y):
        self.model = model
        self.y = y

    def eval_x_first(self, x, gam):
        z = self.diff(x, gam)
        return 0.5 * (z**2).sum()

    def eval_gam_first(self, gam, x):
        return self.eval_x_first(x, gam)

    def diff(self, x, gam):
        # reshaping of x/gam is needed since fmin_l_bfgs_b flattens all arrays
        return (self.model.forward(x.reshape(self.model._image_shape + (2, )),
                                   gam.reshape(self.model._image_shape)) -
                self.y) * self.model.kmask

    def grad_x(self, x, gam):
        in_shape = x.shape
        z = self.diff(x, gam)

        # reshaping of x/gam is needed since fmin_l_bfgs_b flattens all arrays
        return self.model.adjoint(z, gam.reshape(
            self.model._image_shape)).reshape(in_shape)

    def grad_gam(self, gam, x):
        in_shape = gam.shape
        z = self.diff(x, gam)

        # reshaping of x/gam is needed since fmin_l_bfgs_b flattens all arrays
        return self.model.grad_gam(z, gam.reshape(self.model._image_shape),
                                   x.reshape(self.model._image_shape +
                                             (2, ))).reshape(in_shape).copy()


#-------------------------------------------------------------------


class TotalLoss:

    def __init__(self, datafidelityloss, penalty_x, penalty_gam, beta_x,
                 beta_gam):
        self.datafidelityloss = datafidelityloss
        self.penalty_x = penalty_x
        self.penalty_gam = penalty_gam

        self.beta_x = beta_x
        self.beta_gam = beta_gam

        self.x_shape = self.datafidelityloss.model._image_shape + (2, )
        self.gam_shape = self.datafidelityloss.model._image_shape

    def eval_x_first(self, x, gam):
        cost = self.datafidelityloss.eval_x_first(x, gam)

        if self.beta_x > 0:
            # reshaping of x is necessary since LBFGS will pass flattened arrays
            cost += self.beta_x * self.penalty_x.eval(
                x.reshape(self.x_shape)[..., 0])
            cost += self.beta_x * self.penalty_x.eval(
                x.reshape(self.x_shape)[..., 1])

        if self.beta_gam > 0:
            cost += self.beta_gam * self.penalty_gam.eval(gam)

        return cost

    def eval_gam_first(self, gam, x):
        return self.eval_x_first(x, gam)

    def grad_x(self, x, gam):
        # reshaping of x is necessary since LBFGS will pass flattened arrays
        grad = self.datafidelityloss.grad_x(x, gam).reshape(self.x_shape)

        if self.beta_x > 0:
            grad[..., 0] += self.beta_x * self.penalty_x.grad(
                x.reshape(self.x_shape)[..., 0])
            grad[..., 1] += self.beta_x * self.penalty_x.grad(
                x.reshape(self.x_shape)[..., 1])

        return grad.reshape(x.shape)

    def grad_gam(self, gam, x):
        grad = self.datafidelityloss.grad_gam(gam, x)

        if self.beta_gam > 0:
            grad += self.beta_gam * self.penalty_gam.grad(gam)

        return grad
