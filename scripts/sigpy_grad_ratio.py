import sigpy
import numpy as np
from functools import reduce


def setup_operator(ish, coords, rr, nn, delt, sc):
    ops1 = []
    ops2 = []
    # first echo op
    for i, co in enumerate(coords):
        ops1.append(sc * sigpy.linop.NUFFT(ish, co) *
                    sigpy.linop.Multiply(ish, rr**(nn + i * delt)))
        ops2.append(sc * sigpy.linop.NUFFT(ish, co) *
                    sigpy.linop.Multiply(ish, rr**(nn + i * delt + 1)))

    return sigpy.linop.Vstack(ops1), sigpy.linop.Vstack(ops2)


#---------------------------------------------------------------------

np.random.seed(1)

ishape = (3, 2, 2)
num_coords = 4
n = 1.7
delta = 0.4
scale = 1.2

#-------------------------------------------------------------
#-------------------------------------------------------------

# setup the nufft operator
cs = []

for i in range(3):
    coord = np.random.rand(num_coords + 1, len(ishape))
    coord *= ishape[0]
    coord -= ishape[0] / 2

    cs.append(coord)

r = np.random.rand(*ishape)
A_e1, A_e2 = setup_operator(ishape, cs, r, n, delta, scale)
A = sigpy.linop.Vstack([A_e1, A_e2])

# setup random test image
x = np.random.rand(*ishape) + 1j * np.random.rand(*ishape)

# setup random data
y = np.random.rand(*A.oshape) + 1j * np.random.rand(*A.oshape)

# calculate gradient w.r.t ratio image
d = A(x) - y

F0 = []
F1 = []

for i in range(3):
    F0.append(
        sigpy.linop.Multiply(ishape, (n + i * delta) *
                             (r**(n + i * delta - 1)) * x.conj()) *
        sigpy.linop.Compose(A.linops[0].linops[i].linops[:-1]).H)
    F1.append(
        sigpy.linop.Multiply(ishape, (n + i * delta + 1) *
                             (r**(n + i * delta)) * x.conj()) *
        sigpy.linop.Compose(A.linops[1].linops[i].linops[:-1]).H)

F0 = sigpy.linop.Hstack(F0)
F1 = sigpy.linop.Hstack(F1)

H = sigpy.linop.Hstack([F0, F1])

grad = np.real(H(d))

# numerically approximate gradient
c = 0.5 * (d * d.conj()).sum().real

for i in range(x.size):
    eps = 1e-7
    rd = r.copy()
    rd.ravel()[i] += eps
    A2_e1, A2_e2 = setup_operator(ishape, cs, rd, n, delta, scale)
    A2 = sigpy.linop.Vstack([A2_e1, A2_e2])
    d2 = A2(x) - y
    c2 = 0.5 * (d2 * d2.conj()).sum().real

    print(f'{((c2 - c) / eps): .4E} {(grad.ravel()[i]): .4E}')