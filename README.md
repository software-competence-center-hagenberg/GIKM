# GIKM Implementation in Matlab and specified use-cases in Python
This repository is from SCCH for Primetals defining the use-case built by Mohit.

## Installation
Have python pre installed on your computer and define `venv` with

``
python -m venv venv/
``


Install all necessary packages to the virtual environment with `pip install -r requirements.txt`.

## Folder and File Description
- ``delete_files.py``: This file was quick work around to delete unecessary matlab tmp files that have been shared within the initial dataset folder.
- ``Datasets``: This folder is the origin for all the datasets used within the experiments whether Python or Matlab. Since the datasets are not be placed within the Github, they will be provided via an One-Drive downloadable link with the following structure:
  - ./Datasets/
    - CIFAR-100/
    - cifar10/
    - FashionMNIST/
    - FreiburgGrocery/
    - MNIST/
    - office+caltech256/
- ``GIKM``: Official matlab implementation of Mohits paper about [Geometrically Inspired Kernel Machines](https://arxiv.org/abs/2407.04335).
- ``GIKM_python``: Notebooks and python files utilizing python packages to perform GIKM in python.
- ``results``: consisting of .csv files with the performances of the python implementation to compare with the paper.

## File description about GIKM_python
For the GIKM_python there are files preset as ``.py`` and as ``.ipynb``. All the files provded starting with the prefix "run" are the python equivalents to the matlab files from the paper. The general functions needed for the implementation are in the ``func.py``. In the folder there is another file with the name ``optimisedfunc.py``. The ``optimisedfunc.py`` is utilizing a more time efficient implementation of the functions sacrificing accuracy seen in ``run_experiment_mnist_optmised.py``. This file should not be used yet since it is just in early stage experimenting with different timesaving variants. The files dealing with the **office+caltech** use-case are also not ready to be comparted and executed in the python version. The following files are implemented properly in python with the results being there as well:
- ``run_experiment_cifar10.py``
- ``run_experiment_cifar100.py``
- ``run_experiment_fmnist2.py``
- ``run_experiment_mnist.py``
- ``run_experiment_grocery.py``

