# Demosaic ![GitHub top language](https://img.shields.io/github/languages/top/evgkanias/demosaic) [![GitHub license](https://img.shields.io/github/license/evgkanias/demosaic)](https://github.com/evgkanias/demosaic/blob/main/LICENSE) ![GitHub last-commit](https://img.shields.io/github/last-commit/evgkanias/demosaic) [![PyPI](https://github.com/evgkanias/demosaic/actions/workflows/python-package.yml/badge.svg)](https://github.com/evgkanias/demosaic/actions/workflows/python-package.yml) [![Anaconda](https://github.com/evgkanias/demosaic/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/evgkanias/demosaic/actions/workflows/python-package-conda.yml)
A package with functions that help demosaicing images with polarisation and Bayer-RGB or mono filter arrays.
It includes implementations of different algorithms, such as:
* Bilinear interpolation
* Malvar interpolation (Malvar et al., 2024)
* Fourier interpolation (Hagen et al., 2024)

It also contains a module for combining images with different exposures into a single HDR image.




## References
1. Malvar, H. S., He, L.-W. & Cutler, R. High-Quality Linear Interpolation for Demosaicing of Bayer-Patterned Color Images. 2004 IEEE Int. Conf. Acoust., Speech, Signal Process. 3, III-485-III–488 (2004).
2. Hagen, N., Stockmans, T., Otani, Y. & Buranasiri, P. Fourier-domain filtering analysis for color-polarization camera demosaicking. Appl. Opt. 63, 2314 (2024).

## Installation

You can easily install the package by using pip as:
```commandline
pip install git+https://github.com/evgkanias/demosaic.git
```

Alternatively you need to clone the GitHub repository, navigate to the main directory of the project, install the dependencies and finally
the package itself. Here is an example code that installs the package:

1. Clone this repo.
```commandline
mkdir ~/src
cd ~/src
git clone https://github.com/evgkanias/demosaic.git
cd sky
```
2. Install the required libraries. 
   1. using pip :
      ```commandline
      pip install -r requirements.txt
      ```

   2. using conda :
      ```commandline
      conda env create -f environment.yml
      conda activate demosaic-env
      ```
3. Install the package.
   1. using pip :
      ```commandline
      pip install .
      ```
   2. using conda :
      ```commandline
      conda install .
      ```
   
Note that the [pip](https://pypi.org/project/pip/) project might be needed for the above installation.

## Report an issue

If you have any issues installing or using the package, you can report it
[here](https://github.com/evgkanias/demosaic/issues).

## Author

The code is written by [Evripidis Gkanias](https://evgkanias.github.io/).

## Copyright

Copyright &copy; 2026, Evripidis Gkanias; Lund Vision Group; Lund University.