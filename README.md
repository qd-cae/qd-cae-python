

# qd - a CAE Python Library

[![Build Status](https://travis-ci.org/qd-cae/qd-eng.svg?branch=master)](https://travis-ci.org/qd-cae/qd-eng)

This python2 library is an open source software package with utility functions regarding
engineering, especially Simulation and Computer Aided Engineering (CAE).
The library is still young, so features drop in over time depending on the
persons who contribute. Visit our website at http://www.qd-eng.de/

CAE suffers greatly from expensive tools with bad or moderate scripting access.
Also many algorithms developed in research never get published due to licensing
or being very specific regarding a certain software environment. Engineering is
lacking behind here, contrary to modern informatics.

We want to tackle this issue with this free library, in order to give engineering
a boost. Use it, build more awesome applications with it, share your projects
too and if possible contribute.


*Changelog 11.2.2017*
Version 0.4.3

qd.numerics.sampling:
 - changed diversipy method for lhd design

*Changelog 24.2.2017*
Version 0.4.2

qd.cae.dyna
  - part.get_elements now supports element type filtering

qd.numerics.sampling:
  - Added new module
  - added function uniform_lhs (latin hypercube sampling)


# Installation

For **Windows** (x64 python 2.7 or 3.5 and higher) use:

```
pip install qd
```
or
```
python -m pip install qd
```

For **Linux** I have also compiled some wheels in the ```dist``` folder, though they might not work for any distribution. If you need to compile it for yourself, run:

```
git clone https://github.com/qd-cae/qd-eng.git
cd qd-eng
sudo apt-get install python-numpy
sudo apt-get install libboost-dev
pip install --upgrade pip setuptools wheel
pip install --only-binary=numpy numpy
pip install diversipy
python setup.py install
```

**If your distribution is missing please open a request**

If possible, use the pre-compiled python-wheels in the dist folder and install with pip.
The wheel is always compiled with FEMZIP support (not). For compiling the code yourself,
the library needs Boost.

# Descriptions

Current Module List:

1. [qd.cae.dyna](https://github.com/qd-cae/qd/blob/master/QD_CAE_DYNA.md)
  - Read D3plot (DONE)
  - Read Binouts (DONE)
  - Read Input Files (partially done)
2. [qd.numerics](https://github.com/qd-cae/qd/blob/master/QD_NUMERICS.md)
  - Submodule: sampling

For more details, look into the helper files.


# Compilation

If one wants to compile the code himself, register your boost directory in the top of the setup script. If one want's to compile with femzip support, the link libraries need to be downloaded from the official website from SIDACT.

In case one wants to use it for another C++ application, the source code may also be compiled without the python wrapper. CMake is recommended in that case.

# License

See the LICENSE file.
For further licensing requests or questions contact us.

# Authors

- C. Diez
- D. Toewe