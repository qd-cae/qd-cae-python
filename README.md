

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

*Changelog 18.02.2018*
Version 0.7.0

The `KeyFile` class was rewritten entirely with a ton of new features, such as keyword creation and manipulation. The main idea was to create or read a `KeyFile` and preserve its whole structure including comments. We will make a few tutorials to explain enitre the capabilities and limits.

For more info see the [Documentation](https://qd-cae.github.io/qd-cae-python/build/html/index.html) an [Introduction](http://www.qd-eng.de/index.php/2018/02/19/manipulating-ls-dyna-keyfiles-in-python/). We also plan to do more tutorials.


qd.cae.dyna
 - The `KeyFile` class can now read all keywords
   - using `filepath=""` creates an empty `KeyFile` 
   - using `read_keywords=True` reads all keywords as generic keywords
   - using `parse_mesh=True` creates the mesh specific keywords below and loads the mesh. If `parse_mesh=False`, then also the mesh keywords are treated as generic keywords.
   - using `load_includes` ... loads the includes of course
 - Added new classes related to the `KeyFile` enhancement:
   - `Keyword`: Generic keyword class for all keywords
   - `NodeKeyword`: if `parse_mesh=True` then this class is used for nodes keywords
   - `ElementKeyword`: if `parse_mesh=True` then this class is used for element keywords
   - `PartKeyword`: if `parse_mesh=True` then this class is used for part keywords
   - `IncludeKeyword`: handles include keywords
   - `IncludePathKeyword`: handles include path keywords
 - `KeyFile.save` preserves the file structure including comments
 - Note: While one can add mesh entities to a `KeyFile`, one can currently not delete them
 - Note: `KeyFile` reading performance has not been tuned yet ... 
 - Fix: `D3plot` can now read files with temperature field

*Changelog 24.11.2017*
Version 0.6.8

qd.cae.dyna
 - `D3plot.plot` and `Part.plot` function was fixed and works now again
 - few minor improvements and changes in the background


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
