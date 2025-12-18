Code for Teixeira et al. (2024)

This repository contains the code used to produce the results presented in  

Teixeira et al. (2024),
“Photometric Redshifts Probability Density Estimation from Recurrent Neural Networks in the DECam Local Volume Exploration Survey Data Release 2”
Astronomy and Computing
https://doi.org/10.1016/j.ascom.2024.100886

The scope of this repository is restricted to the data processing, modeling, and
analysis steps directly related to the results discussed in the paper.
Components associated with the autoencoder framework described in the work are
intentionally excluded.

------------------------------------------------------------

Repository Structure

.
├── data/              # Example data for running the codes and inspecting data structure
│
├── figures/           # Figures appearing in the paper
│
├── notebooks/         # Example notebooks illustrating the workflow
│
├── scripts/           # Analysis and processing scripts
│
├── utils/             # Custom Python modules used throughout the workflow
│
├── requirements.txt   # Python dependencies

------------------------------------------------------------

Code Overview

scripts/
Scripts responsible for catalog cleaning, cross-matching, model execution,
and generation of the results presented in the paper.

utils/
Shared utility modules, including:
- FITS catalog handling
- catalog matching routines
- photometric quality cuts
- helper functions for parallel processing

notebooks/
Example notebooks demonstrating the main analysis steps and workflow logic.

------------------------------------------------------------

Data

Raw survey data are not included in this repository.

The scripts assume access to external datasets (e.g. DELVE DR2, spectroscopic
reference catalogs). File paths should be adapted by the user within the scripts
or notebooks.

Example expected structure:

data/
├── raw/
├── processed/

------------------------------------------------------------

Requirements

A minimal Python environment can be set up with:

pip install -r requirements.txt

The code has been tested with Python ≥ 3.9.

------------------------------------------------------------

Reproducibility

This repository is intended to support reproducibility of the main results
presented in the paper, assuming access to the corresponding datasets.

For methodological details and scientific context, please refer to the paper.
For questions or issues related to the code, feel free to open an issue in this
repository.
