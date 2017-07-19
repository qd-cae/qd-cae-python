

# qd - a CAE Python Library

[![Build Status](https://travis-ci.org/qd-cae/qd-cae-python.svg?branch=master)](https://travis-ci.org/qd-cae/qd-cae-python)
[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=NLSUW34MEZ2GU)

This Python3 library is an open source software package with utility functions regarding engineering, especially Simulation and Computer Aided Engineering (CAE).
The library is still young, so features drop in over time depending on the needs of the persons who contribute. Visit our website at http://www.qd-eng.de/

CAE suffers greatly from expensive tools with bad or moderate scripting access. Also many algorithms developed in research never get published due to licensing or being very specific regarding a certain software environment. Engineering is lacking behind here in contrast to modern informatics.

We want to tackle this issue with this free library, in order to give engineering a boost. Use it, build more awesome applications with it, share your projects too and if possible contribute.

# What's new

*Changelog 18.05.2017*
Version 0.6.0

The whole C++ Python API was rewritten with pybind11 because the old interface took too much maintenance. The code was also modernized to C++14. Speed is still about the same.

**Note: The API changed a little bit (see below)**

compilation:
  - Boost as dependency was removed from the project (yay)
  - pybind11 was added as dependency (see [compilation](#Compilation))
  - [compilation](#Compilation) in general is now simpler

qd.cae.dyna
  - The whole Python API was rewritten with pybind11 
  - Due to this change, **some new functions** were added (already existed natively)
  - **Element type specifier is not a string anymore**, but uses `Element.type` (enumeration in C++). Use `Element.none`, `Element.beam`, `Element.shell` or `Element.solid` as replacement (might add string support in the future again).
  - `Element.get_history` renamed to `Element.get_history_variables`
  

*Changelog 18.05.2017*
Version 0.5.7

Internal stuff only

# Descriptions

You can find the [full Documentation here](https://qd-cae.github.io/qd-cae-python/build/html/index.html). Brief module and feature list:

1. [qd.cae.dyna ](https://qd-cae.github.io/qd-cae-python/build/html/qd_cae_dyna.html)
  - Read D3plot
  - Read Binouts
  - Read Input Files (partially done)
2. [qd.numerics](https://qd-cae.github.io/qd-cae-python/build/html/qd_numerics.html)
  - Submodule: sampling
3. [qd.cae.beta](https://qd-cae.github.io/qd-cae-python/build/html/qd_cae_beta.html)
  - MetaCommunicator (control META remotely)

# Installation

## Windows

The wheel is always compiled with FEMZIP support. For **Windows** (x64 Python 3.5 and higher). If possible install by using pip. Since the package is uploaded to pipy, the following should work:

```
pip install qd
```
or
```
python3 -m pip install qd
```

The wheels which are used by pip can also be found here in the [dist folder](https://github.com/qd-cae/qd-cae-python/tree/master/dist). Note though that your windows python distribution might not neccessarily find an appropriate version and fail. In that case one has 3 options:

 - Download and use [Anaconda Python x64](https://www.continuum.io/downloads#windows)
 - Open a request for compilation of a specific distribution
 - Compile it yourself (see [Compilation](#Compilation))  

## Linux

In the [dist folder](https://github.com/qd-cae/qd-cae-python/tree/master/dist) usually are some precompiled Linux wheels for installation with pip. Note that Linux wheels rarely work accross different systems, so do not expect them to work and in case of trouble see the [Compilation](#Compilation) section below. It's quite simple to compile it yourself on Linux.

# Compilation

## Windows

Windows only supports compilation for python3.5 and higher. This is for the reason that python2 needs an ancient MSCV compiler, which does not support the new coding standard C++14 (also the old compiler created very very very slow code).

For compilation of python version 3.5 or 3.6 on Windows download [Visual Studio 15](https://www.microsoft.com/de-DE/download/details.aspx?id=48146) (dunno if Visual Studio 17 also works). Then clone the repository into a folder:

```
git clone https://github.com/qd-cae/qd-cae-python.git
cd qd-eng
```

Thereafter run the setup script, the code will automatically compile and install. 

```bash
python3 -m pip install -r requirements.txt
python3 setup.py install
```

If one also wants to compile with FEMZIP support, download the libraries from the [Sidact Homepage](www.sidact.de). There are two things to download, the actual femunzip library, as well as the link libraries (Use FEMZIP 8.68 dyna NO OMP Windows VS2012 MD x64). Put all into a folder and also register the path in the header of the `setup.py`. The installation skips femzip if the libraries are missing (if in doubt then just ignore it).

```python
femzip_path = "path/to/femzip/libs"
```

## Linux

Note that in contrast to windows, on Linux the library can be compiled for both, python2 and python3. Check first whether your compiler is new enough.

```bash
g++ --version
```
You will need **g++ version 4.9 or higher**. If your compiler is fine, just install as follows:

```bash
git clone https://github.com/qd-cae/qd-cae-python.git
cd qd-cae-python
sudo apt-get install python3-numpy
sudo python3 -m pip install -r requirements.txt
sudo python3 setup.py install
```

As a little hint, if you want force python to take your modern compiler version (it may happen you have multiple), simply overwrite the `CC` attribute before the call:

```bash
sudo CC=g++-4.9 python3 setup.py install
```

In case of the desire for FEMZIP support, simply download and register it the same way as for windows.

# License

See the LICENSE file.
For further licensing requests or questions don't hesitate to contact us.

# Authors

- codie (C. Diez)
- towelie (D. Toewe)
