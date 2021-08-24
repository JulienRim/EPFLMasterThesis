# EPFL Master Thesis (2021)
## Modulatory Effects in Neurostimulation from Remote Cortical Areas for Neuroprosthetic Control

---

This github repo contains the functions that I wrote during my the course of my master thesis research. 

These functions require the following libraries
- Numpy
- Pandas
- Matplotlib
- Seaborn
- GPy
- Tdt
- Scipy
- SKLearn
- Itertools
- Collections

---
There are currently 5 python files that contains functions:
- __GP_functions.py:__ contains functions used for Gaussian Process based Bayesian Optimization
- __multi_unit_functions.py:__ contains functions used for threshold crossing and loading the data
- __multi_unit_plotting_functions.py:__ contains functions used for spatial plotting of results. 
- __helpers.py:__ Contains functions meant for loading mat files as dictionaries. This is copied from a stack overflow referenced in the file.
- __single_pulse_functions.py:__ functions that are useful for the LFP dataset. 

There are also two jupyter notebooks that show what some of these functions do and how to use them. 

The data that these functions use is proprietary to the lab, and therefore I cannot share it. 

Last updated: August 24th, 2021