import sigpy
import numpy as np
from utils_sigpy import NUFFTT2starDualEchoModel

#---------------------------------------------------------------------

ishape = (3, 4)
num_readouts = 3
num_samples_per_readout = 1000

fov_cm = 22.

np.random.seed(1)
#-------------------------------------------------------------
#-------------------------------------------------------------

# setup unitless kspace coordinates
coord = np.random.rand(num_samples_per_readout, num_readouts, len(ishape))
coord *= ishape[0]
coord -= ishape[0] / 2

# convert to 1/cm  units
k_1_cm = coord / fov_cm

# setup random test image
x = np.random.rand(*ishape) + 1j * np.random.rand(*ishape) - 0.5 - 0.5j

# setup random ratio image
r = np.random.rand(*ishape)

model = NUFFTT2starDualEchoModel(ishape,
                                 k_1_cm,
                                 field_of_view_cm=fov_cm,
                                 scale=0.03,
                                 acq_sampling_time_ms=0.016,
                                 echo_time_1_ms=0.5,
                                 echo_time_2_ms=5)

A_e1, A_e2 = model.get_operators_w_decay_model(r)

A = sigpy.linop.Vstack([A_e1, A_e2])

# setup random data
data = np.random.rand(*A.oshape) + 1j * np.random.rand(*A.oshape) - 0.5 - 0.5j

# calculate gradient w.r.t ratio image
diff = A(x) - data

# calculate the data fidelity cost
c = 0.5 * (diff * diff.conj()).sum().real

# calculate the gradient w.r.t. to the ratio image
model.x = x
model.data = data
grad = model.data_fidelity_gradient_r(r)

for i in range(x.size):
    eps = 1e-6
    rd = r.copy()
    rd.ravel()[i] += eps
    A2_e1, A2_e2 = model.get_operators_w_decay_model(rd)
    A2 = sigpy.linop.Vstack([A2_e1, A2_e2])
    diff2 = A2(x) - data
    c2 = 0.5 * (diff2 * diff2.conj()).sum().real

    grad_num = (c2 - c) / eps

    abs_diff = np.abs(grad_num - grad.ravel()[i])
    rel_diff = abs_diff / np.abs(grad_num)

    print(
        f'{((c2 - c) / eps): .4E} {(grad.ravel()[i]): .4E} {abs_diff: .4E} {rel_diff: .4E}'
    )
