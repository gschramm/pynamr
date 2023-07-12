from __future__ import annotations
import numpy as np
import h5py

import pydantic


class TPIParameters(pydantic.BaseModel):
    sampling_time_us: float
    num_points: int
    num_cones: int
    num_readouts_per_cone: list[int]
    max_grad_G_cm: float
    gamma_over_2pi_MHz_T: float

    @property
    def max_grad_T_cm(self):
        return self.max_grad_G_cm * 1e-4


def read_grdb(fname: str,
              gamma_over_2pi_MHz_T: float = 11.262,
              verbose: bool = False) -> tuple[np.ndarray, TPIParameters]:
    """read TPI grdb gradient file (header and data) and convert to k-space trajectory

    Parameters
    ----------
    fname : str
        name of the grdb TPI gradient file
    gamma_over_2pi_MHz_T : float, optional
        gyromagnetic ratio (gamma/2pi) in [MHz/T], by default 11.262
    verbose : bool, optional
        verbose output, by default False

    Returns
    -------
    tuple[np.ndarray, TPIGradientParameters]
        k-space trajectory array of shape (num_points, num_readouts)
        (not rotated) in 1/cm,
        TPIGradientParameters object
    """

    # read the file header of the gradient file
    params = np.fromfile(fname, dtype=np.int16, count=6)

    # number of TPI cones
    num_cones = int(params[0])
    # number of points per readount
    num_points = int(params[1])
    # sampling time in us
    dt_us = float(params[2])
    # maximum gradient in G/cm
    max_grad_G_cm = float(params[3]) / 100

    if verbose:
        print(f'num_cones     : {num_cones}')
        print(f'num_points    : {num_points}')
        print(f'dt_us         : {dt_us}')
        print(f'max_grad_G_cm : {max_grad_G_cm}')

    # read the number of readouts for every cone
    num_readouts_per_cone = np.fromfile(fname,
                                        dtype=np.int16,
                                        count=num_cones,
                                        offset=params.itemsize * params.size)

    grads = np.fromfile(
        fname,
        dtype=np.int16,
        count=num_cones * num_points,
        offset=params.itemsize * params.size +
        num_readouts_per_cone.itemsize * num_readouts_per_cone.size).reshape(
            num_cones, num_points).astype(float)

    # put the EOS bit back
    grads[:, -1] += 1

    # scale the gradients (relative to Gmax)
    scale = 32768
    max_grad_T_cm = (1e-4) * max_grad_G_cm

    grads_T_cm = grads * (max_grad_T_cm / scale)
    k_not_rotated = np.cumsum(grads_T_cm,
                              axis=1) * dt_us * gamma_over_2pi_MHz_T

    gradient_parameters = TPIParameters(
        sampling_time_us=dt_us,
        num_points=num_points,
        num_cones=num_cones,
        num_readouts_per_cone=num_readouts_per_cone.tolist(),
        max_grad_G_cm=max_grad_G_cm,
        gamma_over_2pi_MHz_T=gamma_over_2pi_MHz_T)

    return k_not_rotated, gradient_parameters


#-------------------------------------------------------------


def rotate_grdb_tpi_k(kx: np.ndarray, ky: np.ndarray, kz: np.ndarray,
                      num_readouts_per_cone: np.ndarray):

    k = []

    for i, num_readounts in enumerate(num_readouts_per_cone):
        phis = np.linspace(0, 2 * np.pi, num_readounts, endpoint=False)

        tmp = np.zeros((3, phis.shape[0], kx.shape[1]))
        tmp[0, :, :] = np.outer(np.cos(phis), kx[i, :]) - np.outer(
            np.sin(phis), ky[i, :])
        tmp[1, :, :] = np.outer(np.sin(phis), kx[i, :]) + np.outer(
            np.cos(phis), ky[i, :])
        tmp[2, :, :] = np.outer(np.ones_like(phis), kz[i, :])

        k.append(tmp)

    return np.concatenate(k, axis=1)


#-------------------------------------------------------------


def read_tpi_grdb_kspace_trajectory(fname_x: str,
                                    gamma_over_2pi_MHz_T: float = 11.262,
                                    add_lower_halfsphere: bool = True,
                                    output_file: str | None = None,
                                    **kwargs) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    fname_x : str
        name of the gradb x gradient file
    gamma_over_2pi_MHz_T : float, optional
        gyromagnetic ratio (gamma/2pi) in [MHz/T], by default 11.262
    add_lower_halfsphere : bool, optional
        add the lower halfsphere to the k-space trajectory, by default True
    outputfile : str, optional
        name of the output hdf5 file, by default generated from fname_x

    Returns
    -------
    np.ndarray
        k-space trajectory array of shape (num_points, num_readouts, 3)
        in 1/cm
    """

    kx, g_params_x = read_grdb(fname_x,
                               gamma_over_2pi_MHz_T=gamma_over_2pi_MHz_T)
    ky, g_params_y = read_grdb(fname_x.replace('.x.grdb', '.y.grdb'),
                               gamma_over_2pi_MHz_T=gamma_over_2pi_MHz_T)
    kz, g_params_z = read_grdb(fname_x.replace('.x.grdb', '.z.grdb'),
                               gamma_over_2pi_MHz_T=gamma_over_2pi_MHz_T)

    if (g_params_x != g_params_y) or (g_params_x != g_params_z):
        raise ValueError(
            'number of readouts per cone is not the same for all directions')

    k = rotate_grdb_tpi_k(kx, ky, kz, g_params_x.num_readouts_per_cone)

    # transpose to (num_points, num_readouts, 3)
    k = np.transpose(k, (2, 1, 0))

    # the gradient files usually only contain the upper halfsphere
    # of kspace, so we add the lower halfsphere
    if add_lower_halfsphere:
        lower_k = k.copy()
        lower_k[:, :, 2] *= -1
        k = np.concatenate((k, lower_k), axis=1)

    # write the trajectory to hdf5
    if output_file is not None:
        with h5py.File(output_file, 'w') as f:
            dset = f.create_dataset('k',
                                    data=k,
                                    compression="gzip",
                                    compression_opts=9)

            # write the TPI parameters to hdf5 attributes
            for key, value in g_params_x.model_dump().items():
                dset.attrs[key] = value

    return k, g_params_x


#-------------------------------------------------------------

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    fname_x = '/home/georg/Desktop/n28p2dt10g30f23sl.x.grdb'
    output_file = fname_x.replace('.x.grdb', '.h5')

    # read grdb files and write k-space trajectory + header to hdf5
    k, g_params = read_tpi_grdb_kspace_trajectory(fname_x,
                                                  output_file=output_file)

    # restore values from written hdf5 file
    with h5py.File(output_file, 'r') as f:
        k2 = f['k'][...]
        g_params2 = TPIParameters(**f['k'].attrs)

    print(g_params)
    assert (g_params == g_params2)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    step = 20
    for i in range(0, k.shape[1], 17):
        ax.scatter(k[::step, i, 0],
                   k[::step, i, 1],
                   k[::step, i, 2],
                   marker='.',
                   s=1)
    fig.show()