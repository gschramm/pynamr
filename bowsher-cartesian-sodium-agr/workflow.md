1. pre-process data and estimate sensiivities
   ```
   python preprocess_real_data.py TBI-n009
   ``` 
2. create N4 corrected 1H proton MPRAGE in subfolder mprage_proton
   ```
   python n4.py
   ```

3. align N4 corrected 1H proton MPRAGE to sodium data
   ```
   python align_n4.py TBI-n009
   ```

4. create link for aligned 1H MPRAGE in DeNoise_kw0_preprocessed folder
   ```
   ln -s ../mprage_proton/T1_n4_aligned_128.nii.npy t1_coreg_128.npy
   ```

5. run recons
