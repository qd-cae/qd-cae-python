

# qd - a CAE Python Library

[![Build Status](https://travis-ci.org/qd-cae/qd-cae-python.svg?branch=master)](https://travis-ci.org/qd-cae/qd-cae-python)
[![Join the chat at https://gitter.im/sahat/hackathon-starter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/qd-cae-python/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link)

This Python3 library is an open source software package with utility functions regarding engineering, especially Simulation and Computer Aided Engineering (CAE).
The library is still young, so features drop in over time depending on the needs of the persons who contribute. Visit our website at http://www.qd-eng.de/

CAE suffers greatly from expensive tools with bad or moderate scripting access and utility. Also many algorithms developed in research never get published due to licensing or being very specific regarding a certain software environment. Engineering is lacking behind here in contrast to modern informatics.

<p style="text-align: center;"><b>"We don't want to provide another software solution, 
<br>we want, that people can build their own solutions quickly"</b></p>

We believe, that CAE can not evolve, if people do not get comfortable access to the raw data and start doing their own stuff, because 

<p style="text-align: center;"><b>"Creativity originates from freedom and simplicity."</b></p>

This core idea keeps us pushing forward.

# Support this project

[![Become a Patron](https://c5.patreon.com/external/logo/become_a_patron_button.png)](https://www.patreon.com/bePatron?u=8375141)

# What's new?

*Changelog 02.06.2018*
Version 0.8.0

**License: Changed project license to BSD**

qd.cae.dyna:
 - **Arrays** are here: added many functions to get entire data arrays:
   - `d3plot.get_node_displacement(element_filter)`
   - `d3plot.get_node_velocity(element_filter)`
   - `d3plot.get_node_acceleration(element_filter)`
   - `d3plot.get_node_ids(element_filter)`
   - `d3plot.get_element_coords(element_filter)`
   - `d3plot.get_element_energy(element_filter)`
   - `d3plot.get_element_strain(element_filter)`
   - `d3plot.get_element_plastic_strain(element_filter)`
   - `d3plot.get_element_stress(element_filter)`
   - `d3plot.get_element_stress_mises(element_filter)`
   - `d3plot.get_element_history_vars(element_type)`
   - `part.get_node_ids()`
   - `part.get_node_indexes()`
   - `part.get_element_ids(element_filter)`
   - `part.get_element_node_ids(element_type, nNodes)`
 - Getting data arrays from `RawD3plot` does not copy the memory anymore, but numpy instead simply uses the C++ memory (big performance boost).
 - `D3plot` now automatically detects femzip compressed files (argument `use_femzip` is now deprecated).
 - `KeyFile` encryption detection checks now for '-----BEGIN PGP MESSAGE-----' and does not compute the entropy anymore (argument `encryption_detection` is now deprecated).
 - Along with a lot of new features we also introduced a lot of new bugs so that we don't get bored
 

*Changelog 18.04.2018*
Version 0.7.2

This is a bugfix version.

qd.cae.dyna:
 - Fix: Reading a `KeyFile` with `parse_mesh=True` and *SOLID in the new format was skipping lines wrongly


# Documentation

You can find the [full Documentation here](https://qd-cae.github.io/qd-cae-python/build/html/index.html). Brief module and feature list:

1. [qd.cae.dyna ](https://qd-cae.github.io/qd-cae-python/build/html/qd_cae_dyna.html)
  - Read D3plot
  - Read Binouts
  - Read KeyFiles
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
python -m pip install qd
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

See the license file.

# Authors

- codie 
- towelie
- bala
