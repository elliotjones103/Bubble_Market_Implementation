# Modelling and Detecting Asset Price Bubbles Using Option Data

This repository contains the code and supporting data used in my MSc dissertation on modelling and detecting asset price bubbles.

The project focuses on using option market data and numerical analysis to study bubble behaviour, estimate martingale defects, and generate the tables, plots, and figures used in the dissertation. The empirical case study in this repository uses Nvidia option data across multiple maturities.

## Overview

This repository was used as the main computational workspace for my dissertation. It includes:

- a Jupyter notebook containing the core analysis workflow
- cleaned and quoted NVDA option datasets across several expiries
- code used to generate dissertation figures, tables, and numerical outputs

The wider aim of the project was to investigate whether option prices can reveal evidence of asset price bubbles and to explore how bubble related quantities can be estimated in practice.

## Main Files

### `DISS SUB (1).ipynb`

This is the main notebook for the dissertation analysis. It contains the core code used for:

- importing and cleaning option market data
- running numerical experiments related to asset price bubbles
- estimating quantities from the modelling framework
- producing plots, tables, and figures for the dissertation
- testing option based bubble detection ideas

### NVDA option data files

The CSV files contain Nvidia option data for different expiry dates and are used as inputs for the empirical analysis.

- files labelled **CLEAN** contain processed or filtered datasets
- files labelled **QUOTES** contain quote level market data used in the analysis pipeline

## What This Repository Was Used For

This codebase was used to support the dissertation by:

- cleaning and preparing option market data
- analysing cross maturity option information
- exploring numerical evidence of bubble behaviour
- estimating bubble related quantities from option prices
- generating the computational results presented in the final dissertation

## Reproducibility

To run the analysis:

1. Clone or download the repository
2. Open the notebook in Jupyter Notebook or JupyterLab
3. Make sure the CSV files remain in the same relative location as the notebook
4. Run the notebook cells in order

## Suggested Python Libraries

The project uses a standard scientific Python stack, likely including:

- `pandas`
- `numpy`
- `matplotlib`
- `scipy`
- `jupyter`

You can install these manually in your Python environment before running the notebook.

## Research Context

This repository supports my dissertation research on financial bubbles, with a particular focus on modelling and detecting bubble effects through option market data.

The code is intended for academic research purposes and forms the computational backbone of the dissertation results.

## Notes

- this repository was created to accompany my dissertation rather than as a polished software package
- file names have been kept close to the working dissertation version used during the project
- the notebook and datasets are included so that the workflow can be inspected and reproduced

## Author

**Elliot Jones**  
MSc Financial Mathematics with Data Science  
University of Bath
