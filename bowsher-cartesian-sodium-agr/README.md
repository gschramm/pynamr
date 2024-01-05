Workflow for running Siemens birdcage sodium AGR
================================================

Setup / Install
---------------

1. Install conda or mamba, e.g. from https://github.com/conda-forge/miniforge

2. Setup a new virtual environment that contains all dependencies
   ```
   mamba env create -f environment.yml     (or, but slower)
   conda env create -f environment.yml
   ```

3. Activate the new env. 
   ```
   conda activate bowsher-cartesian-sodium-agr
   ```

4. run the recons as described below

Running recons
--------------

1. configure input file names in a config.json file (start from demo_config.json)
     in the config.json you have to specify:
       - "TE05c0_filename": abs. path to the TE05 c0 sodium image you want to use (kw0 recommended)
       - "TE5c0_filename": abs. path to the TE5 c0 sodium image you want to use (kw0 recommended)
       - "TE05soskw0_filename": abs. path to the TE05 kw0 sos file (for export only)
       - "TE05soskw1_filename": abs. path to the TE05 kw1 sos file (for export only)
       - "MPRAGE_filename": abs. path of the MPRAGE "raw" float image 

2. preprocess the sodium data, run N4 of the proton MPRAGE and align the proton MPRAGE
   ```
   python preprocess.py --cfg my_config.json
   ``` 
3. run 3 AGRs with different beta values (pre-defined in the recon parameter jsons)
   ```
   python sodium_bowsher_agr.py --cfg my_config.json --recon_cfg recon1_params1.json
   python sodium_bowsher_agr.py --cfg my_config.json --recon_cfg recon1_params2.json
   python sodium_bowsher_agr.py --cfg my_config.json --recon_cfg recon1_params3.json
   ```