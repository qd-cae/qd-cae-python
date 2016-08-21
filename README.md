

# QD-Engineering 4 Python

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

# Sub-Modules

Current Module List:

1. [qd.cae.dyna](https://github.com/qd-cae/qd/blob/master/QD_CAE_DYNA.md)
  - Read D3plot (DONE)
  - Read Input Files (IN PPROGRESS)
  - Read Binouts (TODO)
2. qd.numerics
  - certain numerical functions missing in numpy or scipy

For more details, look into the helper files.

# Installation

If possible, use the pre-compiled python-wheels in the dist folder and install with pip.
The wheel is always compiled with FEMZIP support. For compiling the code yourself,
the library needs Boost.

The library is written almost entirely in C++, thus compilation is required if
triggering the setup script. There are two dependencies, from which one is
optional. BOOST and FEMZIP. If one wants to compile with FEMZIP, the link libraries
need to be downloaded from the official website. Also we use some boost headers,
so boost is not optional. It is not uploaded here, due to it's large size.

In case one wants to use it for another C++ application, the source code may
also be compiled without the python2 wrapper.

# License

See the LICENSE file.
For further licensing requests or questions contact us.

# Authors

- C. Diez
- N. Klinke
