# Depth Mapping in Turbid and Deep Waters Using AVIRIS-NG

This repository contains the source code, a matched dataset of depth and reflectance spectra from AVIRIS-NG, and the RF-RFE model pickle file. These resources were utilized for the study "Depth Mapping in Turbid and Deep Waters Using AVIRIS-NG Imagery: A Study in Wax Lake Delta, Louisiana, USA" by Kwon et al., which is currently in preparation.

![github_fig](https://github.com/ksy92/Hmap_AVIRIS-NG/assets/35686126/c6f008ed-5d91-4266-b782-1c7a0fe7e7de)

## Input Files

The files `H_Rrs_df2021a_window_3.csv` and `H_Rrs_df2021b_window_3.csv` are matched datasets consisting of depth and reflectance spectra. These datasets were created using a sliding window-based pixel averaging technique during the DELTA-X spring and fall campaigns, respectively.

## Code Description

The script `WLD_DEPTH_MAPPING_REGRESSION.py` includes the processes for training, testing, and band selection using Recursive Feature Elimination with Cross-Validation (RFE-CV) for both Partial Least Squares Regression (PLSR) and Random Forest (RF) regressors. The script `WLD_DEPTH_MAPPING_FUNCTIONS.py` provides essential functions required to run the regression code.

## Acknowledgments and Disclaimer

The source code was developed by Siyoon Kwon. This work has been supported by the NASA Delta-X project, which is funded by the Science Mission Directorate's Earth Science Division through the Earth Venture Suborbital-3 Program (NNH17ZDA001
