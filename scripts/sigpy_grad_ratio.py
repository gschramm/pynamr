import sigpy
import numpy as np
from functools import reduce


def setup_operator(ish, co1, co2, rr, nn1, nn2, sc):
    op1 = sc * sigpy.linop.NUFFT(ish, co1) * sigpy.linop.Multiply(
        ishape, rr**nn1)
    op2 = sc * sigpy.linop.NUFFT(ish, co2) * sigpy.linop.Multiply(
        ishape, rr**nn2)

    return sigpy.linop.Vstack([op1, op2])


#---------------------------------------------------------------------

np.random.seed(1)

ishape = (4, )
num_coords = 6
n1 = 0.7
n2 = n1 + 1
scale = 1.2

#-------------------------------------------------------------
#-------------------------------------------------------------

# setup the nufft operator
coord1 = np.random.rand(num_coords, len(ishape))
coord1 *= ishape[0]
coord1 -= ishape[0] / 2

coord2 = np.random.rand(num_coords + 1, len(ishape))
coord2 *= ishape[0]
coord2 -= ishape[0] / 2

r = np.random.rand(*ishape)
A = setup_operator(ishape, coord1, coord2, r, n1, n2, scale)

# setup random test image
x = np.random.rand(*ishape) + 1j * np.random.rand(*ishape)

# setup random data
y = np.random.rand(*A.oshape) + 1j * np.random.rand(*A.oshape)

# calculate gradient w.r.t ratio image
d = A(x) - y

F1 = sigpy.linop.Multiply(ishape,
                          n1 * (r**(n1 - 1)) * x.conj()) * sigpy.linop.Compose(
                              A.linops[0].linops[:-1]).H
F2 = sigpy.linop.Multiply(ishape,
                          n2 * (r**(n2 - 1)) * x.conj()) * sigpy.linop.Compose(
                              A.linops[1].linops[:-1]).H

H = sigpy.linop.Hstack([F1, F2])

grad = np.real(H(d))

# numerically approximate gradient
c = 0.5 * (d * d.conj()).sum().real

for i in range(ishape[0]):
    eps = 1e-7
    rd = r.copy()
    rd[i] += eps
    A2 = setup_operator(ishape, coord1, coord2, rd, n1, n2, scale)
    d2 = A2(x) - y
    c2 = 0.5 * (d2 * d2.conj()).sum().real

    print((c2 - c) / eps, grad[i])