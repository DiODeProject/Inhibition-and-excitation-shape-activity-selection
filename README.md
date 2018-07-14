# Inhibition-and-excitation-shape-activity-selection
This repository contains code to generate and analyse simulated data that is used in the study of "Inhibition and excitation shape activity selection: effect of oscillations in a decision-making circuit".

Simulations and data analysis were performed in Python code using Jupyter notebooks. For more information on Jupyter notebooks including installation instructions please visit http://jupyter.org/. 

A major part of the notebooks provided uses `ipyparallel` to run calculations in parallel in a standard Python 3 environment.
Information on how to install and use the `ipyparallel` package can be found following this link https://ipyparallel.readthedocs.io/en/latest/.

All code to generate data is available in this repository. However, some of the calculations take a significant amount of time (~ several hours or days). Therefore we also provided the data files produced by the code which is placed in subfolders called `DataGeneration`. Notebooks to produce all graphics is also included. Images will be plotted inside the notebooks and may be exported using standard Matplotlib functionality.

Bifurcation diagrams may also be reproduced. Please read the `Readme_Bifurcation` in the folder `BifurcationDiagrams` for further information. 
