

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


*Changelog 03.03.2017*
Version 0.4.4

 qd.cae.beta
  - added new module (for Beta CAE Systems)
  - added class MetaCommunicator

*Changelog 11.2.2017*
Version 0.4.3

 qd.cae.dyna
  - fixed elout in Binout (was wrong)

# Descriptions

Current Module List (CLICK ON MODULES FOR DOCUMENTATION!):

1. [qd.cae.dyna ](https://github.com/qd-cae/qd/blob/master/QD_CAE_DYNA.md)
  - Read D3plot (DONE)
  - Read Binouts (DONE)
  - Read Input Files (partially done)
2. [qd.numerics](https://github.com/qd-cae/qd/blob/master/QD_NUMERICS.md)
  - Submodule: sampling
3. [qd.cae.beta](https://github.com/qd-cae/qd/blob/master/QD_CAE_BETA.md)
  - MetaCommunicator (control META remotely)

For more details, look into the helper files.

# Installation

If possible, use the pre-compiled python-wheels in the dist folder and install with pip. The wheel is always compiled with FEMZIP support. For **Windows** (x64 python 2.7 or 3.5 and higher) use:

```
pip install qd
```
or
```
python -m pip install qd
```

Note though that the precompiled .whl might not work for all distributions.  **If your distribution is missing please open a request**

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

 For compiling the code yourself on windows or linux, see further below.

# Compilation

For compilation on ANY platform first download [BOOST for c++](https://github.com/boostorg/boost) (we only need headers, do not compile it). On Linux (here Ubuntu) one can just use the package manager:

```
sudo apt-get install libboost-dev
```

For compilation on WINDOWS download either [Visual C++ Python Compiler for Python 2.7)[https://www.microsoft.com/en-us/download/details.aspx?id=44266] or [Visual Studio 15 for Python 3.5 and 3.6](https://www.microsoft.com/de-DE/download/details.aspx?id=48146) (dunno if Visual Studio 17 also works). Then clone the repository into a folder:
```
git clone https://github.com/qd-cae/qd-eng.git
```
Now register the boost directory in the header of the file: setup.py . Thereafter run the setup script, the code will automatically compile and install.

```
python setup.py install
```

On LINUX either install boost as described above manually or by package manager. In case of the package manager, the setup script should automatically find boost on your system and you do not need to register it manually. Just follow the commands in the installation section. 

# License

See the LICENSE file.
For further licensing requests or questions contact us.

# Authors

- C. Diez
- D. Toewe
