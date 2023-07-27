# Anatomically-guided reconstruction (AGR) of dual echo sodium TPI including T2\* decay estimation and modeling

### Installation

Install conda (or mamba). All dependencies that are needed to run the reconstructions are available on conda-forge.

Create a new virtual conda environment that contains all our dependencies

```
conda env create -f environment.yaml
```

**Since reconstructions are run on CUDA GPUs, the dependencies (cupy) can only be installed on systems with a CUDA capable GPU.**

---

### Running recontructions

Activate the new environment

```
conda activate sodium_mr_agr
```

Reconstruction can be run using the `sodium_dual_echo_agr.py` master script. To see all the command line options use:

```
python sodium_dual_echo_agr.py -h
```

**The top section of `sodium_dual_echo_agr.py` explains the expected layout of the input data.**

---

### Data preprocessing

To preprocess raw non-uniform kspace data into the expected hdf5 file format, use the functions defined in the `preprocessing` module.
See the top section of `sodium_dual_echo_agr.py` for me details.

**Structure of kspace_trajectory.h5**

The hdf5 file containing the coordinates of the kspace trajectory
should have the following structure.
It mainly consists of the dataset `k` containing an array of shape
`(num_time_points, num_readouts, 3)`.

```
HDF5 "kspace_trajectory.h5" {
GROUP "/" {
   DATASET "k" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( 3633, 1596, 3 ) / ( 3633, 1596, 3 ) }
      ATTRIBUTE "gamma_over_2pi_MHz_T" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SCALAR
         DATA {
         (0): 11.262
         }
      }
      ATTRIBUTE "max_grad_G_cm" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SCALAR
         DATA {
         (0): 0.16
         }
      }
      ATTRIBUTE "num_cones" {
         DATATYPE  H5T_STD_I64LE
         DATASPACE  SCALAR
         DATA {
         (0): 28
         }
      }
      ATTRIBUTE "num_points" {
         DATATYPE  H5T_STD_I64LE
         DATASPACE  SCALAR
         DATA {
         (0): 3633
         }
      }
      ATTRIBUTE "num_readouts_per_cone" {
         DATATYPE  H5T_STD_I64LE
         DATASPACE  SIMPLE { ( 28 ) / ( 28 ) }
         DATA {
         (0): 90, 88, 88, 88, 86, 86, 84, 82, 80, 78, 74, 72, 68, 64, 62, 58,
         (16): 54, 50, 46, 40, 36, 32, 28, 22, 18, 12, 8, 2
         }
      }
      ATTRIBUTE "sampling_time_us" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SCALAR
         DATA {
         (0): 10
         }
      }
   }
}
}
```

**Structure of converted_data.h5**

The structure of the converted raw data files is listed below.
It mainly contains an array of shape (num_coil_channels, num_time_points, num_readouts).

```
HDF5 "converted_data.h5" {
GROUP "/" {
   DATASET "data" {
      DATATYPE  H5T_COMPOUND {
         H5T_IEEE_F32LE "r";
         H5T_IEEE_F32LE "i";
      }
      DATASPACE  SIMPLE { ( 1, 3632, 1596 ) / ( 1, 3632, 1596 ) }
      ATTRIBUTE "hdr_files" {
         DATATYPE  H5T_STRING {
            STRSIZE H5T_VARIABLE;
            STRPAD H5T_STR_NULLTERM;
            CSET H5T_CSET_UTF8;
            CTYPE H5T_C_S1;
         }
         DATASPACE  SIMPLE { ( 1 ) / ( 1 ) }
         DATA {
         (0): "meas_64759.pwr.0.hdr"
         }
      }
   }
}
}
```
