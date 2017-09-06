

# qd - a CAE Python Library

[![Build Status](https://travis-ci.org/qd-cae/qd-cae-python.svg?branch=master)](https://travis-ci.org/qd-cae/qd-cae-python)
[![Join the chat at https://gitter.im/sahat/hackathon-starter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/qd-cae-python/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link)

This Python3 library is an open source software package with utility functions regarding engineering, especially Simulation and Computer Aided Engineering (CAE).
The library is still young, so features drop in over time depending on the needs of the persons who contribute. Visit our website at http://www.qd-eng.de/

CAE suffers greatly from expensive tools with bad or moderate scripting access and utility. Also many algorithms developed in research never get published due to licensing or being very specific regarding a certain software environment. Engineering is lacking behind here in contrast to modern informatics.

We want to tackle this issue with this free library, in order to give engineering a boost. Simple data access is crucial for further ideas to emerge! Use this library, build more awesome applications with it, share your projects too if not critical and please contribute to us or science in general.

# What's new?

*Changelog 27.08.2017*
Version 0.6.2

qd.cae.dyna
 - D3plot supports now thick shells
 - D3plot can now read files with airbags (does not read the airbags itself!)

Improved code readibility.

*Changelog 24.08.2017*
Version 0.6.1

Improved documentation.

qd.cae.dyna
 - fixed an issue, that occasionally the part ids were wrong

*Changelog 28.07.2017*
Version 0.6.0

**The API changed a little bit (see below)**

The whole C++ Python API was rewritten with pybind11 because the old interface took too much maintenance. This change makes it easier to implement further developmens. The code was also modernized to C++14. Speed is slightly better. 

Note that the API changed a little bit, which may happen for a software with a version below 1.0.0! This will not be done often, so don't worry.

[compilation](#Compilation) in general is now simpler:
  - Boost as dependency was removed from the project (yay)
  - pybind11 was added, but is installed via requirements.txt


qd.cae.dyna
  - The whole Python API was rewritten with pybind11
  - **API Changes**:
    - **Element type specifier is not a string anymore**, but uses `Element.type` (enumeration in C++). Use `Element.none`, `Element.beam`, `Element.shell` or `Element.solid` as replacement (might add string support in the future again).
    - `Element.get_history` renamed to `Element.get_history_variables`
  - **New functions** were added (already existed natively)
    - `femfile.get_partByIndex`
    - `femfile.get_partByName`
    - `femfile.get_elementByIndex`
    - `d3plot.get_nTimesteps`
    - `d3plot.get_title`
  

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

For **Windows** (x64 Python 3.5 and higher) install by using pip. The precompiled wheels are always compiled with FEMZIP support, allowing you to read compressed files directly. Since the package is uploaded to pipy, the following should work:

```
pip install qd
```

or

```
python3 -m pip install qd
```

The wheels, which are used by pip, can also be found here in the [dist folder](https://github.com/qd-cae/qd-cae-python/tree/master/dist). If neccessary download the appropriate one and install it with:

```python
pip install [path/to/downloaded/wheel]
```

Note though that your windows python distribution might not neccessarily find an appropriate version and fail. In that case one has three options:

 - Download and use [Anaconda Python x64](https://www.continuum.io/downloads#windows)
 - Compile it yourself (see the [compilation guide](https://qd-cae.github.io/qd-cae-python/build/html/compilation_guide.html))  
 - Open a request for compilation of a specific distribution

## Linux

In the [dist folder](https://github.com/qd-cae/qd-cae-python/tree/master/dist) usually are some precompiled Linux wheels for installation with pip. Note that Linux wheels rarely work accross different systems, so do not expect them to work and in case of trouble see the [compilation guide](https://qd-cae.github.io/qd-cae-python/build/html/compilation_guide.html). It's quite simple to compile it yourself on Linux (❤).

# Compilation

If you are bold enough to compile it yourself, then follow the [compilation guide](https://qd-cae.github.io/qd-cae-python/build/html/compilation_guide.html).

# License

See the LICENSE file.
For further licensing requests or questions don't hesitate to contact us.

# Authors

- codie (C. Diez)
- towelie (D. Toewe)
