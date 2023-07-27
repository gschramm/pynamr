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
python sodium_dual_echo_agr.py
```

**The top section of `sodium_dual_echo_agr.py` explains the expected layout of the input data.**

---

### Data preprocessing

To preprocess raw non-uniform kspace data into the expected hdf5 file format, use the functions defined in the `preprocessing` module.
See the top section of `sodium_dual_echo_agr.py` for me details.
