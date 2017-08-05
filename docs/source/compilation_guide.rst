
Compilation Guide
=================

Compilation of the package is quite simple (said the expert ...).


Compilation on Windows
----------------------

Windows only supports compilation for python3.5 and higher. This is for the reason that python2 needs an ancient MSCV compiler, which does not support the new coding standard C++14 (also the old compiler created 3x slower code).

For compilation of python version 3.5 or 3.6 on Windows download `Visual Studio 15`_ (dunno if Visual Studio 17 also works, but may be worth a try). Then clone the repository into a folder:

.. _Visual Studio 15: https://www.visualstudio.com/de/vs/older-downloads/

.. code-block:: bash

    git clone https://github.com/qd-cae/qd-cae-python.git
    cd qd-cae-python

Thereafter run the setup script, the code will automatically compile and install. 

.. code-block:: bash
    python3 -m pip install -r requirements.txt
    python3 setup.py install



Compilation on Linux
--------------------

Note that in contrast to windows, on Linux the library can be compiled for both, python2 and python3. Check first whether your compiler is new enough.

.. code-block:: bash

    g++ --version

You will need **g++ version 4.9 or higher**. If your compiler is fine, just install as follows:

.. code-block:: bash

    git clone https://github.com/qd-cae/qd-cae-python.git
    cd qd-cae-python
    sudo python3 -m pip install -r requirements.txt
    sudo python3 setup.py install

As a little hint, if you want force python to take your modern compiler version (it may happen that you have multiple g++), simply overwrite the `CC` attribute before the running the setup script:

.. code-block:: bash

    sudo CC=g++-4.9 python3 setup.py install


Compilation with FEMZIP support
-------------------------------

Important note: The installation skips femzip if the libraries are missing (checks if folder with link libraries exists). If it can not find the folder, then the compiler definition :code:`QD_USE_FEMZIP` is not used. This cancels out all code in the source files regarding FEMZIP.

If one wants to compile with FEMZIP support, download the libraries from the `Sidact Homepage`_. There are two things to download, the actual femunzip library:

 - Windows: **FEMZIP_[version]_dyna_NO_OMP_Windows_VS2012_MD_x64.tar.gz**
 - Linux: **FEMZIP_[version]_dyna_NO_OMP_Linux_64Bit.tar.gz**

as well as the link libraries (**FEMUNZIPLIB-LPR_[version]_link_libraries.tar.gz**). Put all content into a folder and register the path in the header of the `setup.py`. 

.. code-block:: python

    femzip_path_windows = "path/to/femzip/windows/libs"
    femzip_path_linux   = "path/to/femzip/linux/libs"



.. _Sidact Homepage: http://www.sidact.com/

