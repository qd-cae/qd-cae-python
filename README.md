

# qd - a CAE Python Library

[![Build Status](https://travis-ci.org/qd-cae/qd-eng.svg?branch=master)](https://travis-ci.org/qd-cae/qd-eng)

This **Python3** library is an open source software package with utility functions regarding
engineering, especially Simulation and Computer Aided Engineering (CAE).
The library is still young, so features drop in over time depending on the
persons who contribute. Visit our website at http://www.qd-eng.de/

CAE suffers greatly from expensive tools with bad or moderate scripting access.
Also many algorithms developed in research never get published due to licensing
or being very specific regarding a certain software environment. Engineering is
lacking behind here in contrast to modern informatics.

We want to tackle this issue with this free library, in order to give engineering
a boost. Use it, build more awesome applications with it, share your projects
too and if possible contribute.

*Changelog 22.03.2017*
Version 0.5.2

qd.cae.dyna
 - added mises stress as element result

*Changelog 20.03.2017*
Version 0.5.1

qd.cae.dyna
 - fixed: reading of state results for mattyp!=0
 - fixed: solids stresses were not read, if no shell stresses were present

*Changelog 19.03.2017*
Version 0.5.0

General:
 - switched to **PYTHON3 ONLY** (much faster now), but windows does not support python2 anymore.

qd.cae.dyna
 - many many performance improvements
 - Added D3plot.clear for removing already read data
 - D3plot can now handle rigid shell elements (mattyp != 0)
 - Element, Node and Part equals operator is now correct


# Descriptions

Current Module List (**CLICK on MODULES for full DOCUMENTATION!**, Wiki coming soon):

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

## Windows

If possible, use the pre-compiled python-wheels in the dist folder and install with pip. The wheel is always compiled with FEMZIP support. For **Windows** (x64 Python 3.5 and higher) use:

```
pip install qd
```
or
```
python3 -m pip install qd
```

Note though that your python distribution might not neccessarily find an appropriate version and fail. One now has 3 options:

 - Download and use [Anaconda Python x64](https://www.continuum.io/downloads#windows)
 - Open a request for compilation of a specific distribution
 - Compile it yourself (see [Compilation](#Compilation))  

## Linux

In the [dist folder](https://github.com/qd-cae/qd-eng/tree/master/dist) usually are some precompiled Linux wheels for installation with pip. Note that Linux wheels rarely work accross different systems, so do not expect them to work and also please see the [Compilation](#Compilation) section below. It's quite simple to compile it yourself.

# Compilation

## Windows

Windows only supports compilation for python3.5 and higher. This is for the reason that python2 needs an ancient MSCV compiler, which does not support the new coding standard C++11.

First download [BOOST for C++](http://www.boost.org/) (we only need headers, do not compile it). 

For compilation of python version 3.5 or 3.6 on Windows download [Visual Studio 15](https://www.microsoft.com/de-DE/download/details.aspx?id=48146) (dunno if Visual Studio 17 also works). Then clone the repository into a folder:

```
git clone https://github.com/qd-cae/qd-eng.git
cd qd-eng
```

Now register the boost directory in the header of the file: setup.py. There is a variable for it:

```python
# In top of setup.py
boost_path = "path/to/boost"
```

Thereafter run the setup script, the code will automatically compile and install. 

```
python3 setup.py install
```

If one also wants to compile with FEMZIP support, download the libraries from the [Sidact Homepage](www.sidact.de). There are two things to download, the actual femunzip library, as well as the link libraries (Use FEMZIP 8.68 dyna NO OMP Windows VS2012 MD x64). Put all into a folder and also register the path in the header of the setup.py

```python
femzip_path = "path/to/femzip/libs"
```

## Linux

Note that in contrast to windows, on Linux the library can be compiled for both, python2 and python3. 

On Linux (here Ubuntu) one can just use the package manager to install boost

```bash
sudo apt-get install libboost-dev
```

The setup script should now automatically find boost on your system, if you still have trouble download and register it the same way as on windows. Now just install with

```
git clone https://github.com/qd-cae/qd-eng.git
cd qd-eng
sudo apt-get install python3-numpy
sudo python3 -m pip install --upgrade pip setuptools wheel
sudo python3 -m pip install --only-binary=numpy numpy
sudo python3 -m pip install diversipy
sudo python3 setup.py install
```

In case of the desire for FEMZIP support, simply download and register it the same way as for windows.

# License

See the LICENSE file.
For further licensing requests or questions contact us.

# Authors

- C. Diez
- D. Toewe
