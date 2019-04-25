# Inhibition-and-excitation-shape-activity-selection

### Simulation code provided in this repository has been developed by Thomas Bose as part of the DiODe project (https://diode.group.shef.ac.uk/) and is open source. It may be used and modified for non-commercial use. If used without significant modification please cite: [T. Bose, A. Reina & J.A.R. Marshall (2019), *Neural Computation* **31**, 870-896](https://doi.org/10.1162/neco_a_01185). 

### For further information please contact t.bose@sheffield.ac.uk.

This repository contains code to generate and analyse simulated data that is used in the study of **Inhibition and excitation shape activity selection: effect of oscillations in a decision-making circuit** which has been accepted for publication in *Neural Computation*.

Simulations and data analysis were performed in Python code (Python 3) using Jupyter notebooks. For more information on Jupyter notebooks including installation instructions please visit http://jupyter.org/. 

A major part of the notebooks provided uses `ipyparallel` to run calculations in parallel in a standard Python 3 environment.
Information on how to install and use the `ipyparallel` package can be found following this link https://ipyparallel.readthedocs.io/en/latest/.

All code to generate data is available in this repository. However, some of the calculations take a significant amount of time (~ several hours or days). Therefore we also provided the data files produced by the code which are placed in subfolders called `DataGeneration`. Notebooks to produce all graphics are also included. Images will be plotted inside the notebooks and may be exported using standard Matplotlib functionality.

Bifurcation diagrams may also be reproduced. Please read the `Readme_Bifurcation` file in the folder `BifurcationDiagrams` for further information. 
