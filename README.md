

# qd - a CAE Python Library

[![Build Status](https://travis-ci.org/qd-cae/qd-cae-python.svg?branch=master)](https://travis-ci.org/qd-cae/qd-cae-python)
[![Join the chat at https://gitter.im/sahat/hackathon-starter](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/qd-cae-python/Lobby?utm_source=share-link&utm_medium=link&utm_campaign=share-link)

This Python3 library is an open source software package with utility functions regarding engineering, especially Simulation and Computer Aided Engineering (CAE).
The library is still young, so features drop in over time depending on the needs of the persons who contribute. Visit our website at http://www.qd-eng.de/

CAE suffers greatly from expensive tools with bad or moderate scripting access and utility. Also many algorithms developed in research never get published due to licensing or being very specific regarding a certain software environment. Engineering is lacking behind here in contrast to modern informatics.

<p style="text-align: center;"><b>"We don't want to provide another software solution, 
<br>we want, that people can build their own solutions quickly"</b></p>

We believe, that CAE can not evolve, if people do not get comfortable access to the raw data and start doing their own stuff, because 

<p style="text-align: center;"><b>"Creativity originates from freedom and thus simplicity."</b></p>

This core idea keeps us pushing forward.

# Support this project

[![Become a Patron](https://c5.patreon.com/external/logo/become_a_patron_button.png)](https://www.patreon.com/bePatron?u=8375141)

# What's new?

*Changelog 03.04.2018*
Version 0.7.2

qd.cae.dyna 🐰
 - Parallelized many internal functions 🐇 This speeds up running scripts opnening result files chached by the OS (e.g. rerunning a script multiple times).
 - Compilation on linux now requires at least g++-5.4 due to newer features 🥚
 - Added more [Youtube Videos](https://youtu.be/O_vaW3dlJg4) to the documentation

*Changelog 07.03.2018*
Version 0.7.1

This is a bugfix version.

qd.cae.dyna:
  - Fix: loading includes manually could lead to a bug 
  - Fix: if `load_includes=False`, includes are now treated as generic keywords 
  - Fix: add_part could cause a segfault
  - Fix: `KeyFile.save` did only write one keyword, if they had the same `KeyFile.position``.
  - Fix: Adding a `Keyword` without a `position` specified simply appends the `Keyword` to the end of the file.


# Documentation

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

*This library is a free gift from and to the community. We publish under GNU GPL v3 (see the LICENSE file), because we want to encourage everyone to share their code as well. As long as one is not selling their code, in which is library is embedded, we will not demand the disclosure though. Against financial compensation we can also provide professional support or a different license.*

# Authors

- codie (C. Diez)
- towelie (D. Toewe)
- (?)
