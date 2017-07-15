#!/usr/bin/env python

import os
import sys
import glob
import platform
import numpy as np
import unittest
import pybind11
from setuptools import setup, Extension

# ======= S E T T I N G S ======= #
pybind11_path = pybind11.get_include()
femzip_path = "libs/femzip/FEMZIP_8.68_dyna_NO_OMP_Windows_VS2012_MD_x64/x64"  # optional
# femzip_path = "libs/femzip/Linux/64Bit" # optional
# ====== D E V E L O P E R ====== #
debugging_mode = False
measure_time = False
_version = "0.6.0"
# =============================== #


# (1) Native Code Stuff
# (1.1) DYNA-CPP toolbox

compiler_args_dyna = []
include_dirs_dyna = [np.get_include(), "qd/cae", pybind11_path]
lib_dirs_dyna = []
libs_dyna = []
srcs_dyna = [
    "qd/cae/dyna_cpp/python_api/pybind_wrapper.cpp",
    "qd/cae/dyna_cpp/db/FEMFile.cpp",
    "qd/cae/dyna_cpp/db/DB_Elements.cpp",
    "qd/cae/dyna_cpp/db/DB_Nodes.cpp",
    "qd/cae/dyna_cpp/db/DB_Parts.cpp",
    "qd/cae/dyna_cpp/db/Element.cpp",
    "qd/cae/dyna_cpp/db/Node.cpp",
    "qd/cae/dyna_cpp/db/Part.cpp",
    "qd/cae/dyna_cpp/dyna/D3plotBuffer.cpp",
    "qd/cae/dyna_cpp/dyna/D3plot.cpp",
    "qd/cae/dyna_cpp/dyna/KeyFile.cpp",
    "qd/cae/dyna_cpp/dyna/DynaKeyword.cpp",
    "qd/cae/dyna_cpp/utility/FileUtility.cpp"]

# FEMZIP usage? Libraries present?
# You need to download the femzip libraries yourself from SIDACT GmbH
# www.sidact.de
# If you have questions, write a mail.
if os.path.isdir(femzip_path):
    if (platform.system() == "Windows") and os.path.isdir(os.path.join(femzip_path)):
        srcs_dyna.append("qd/cae/dyna_cpp/dyna/FemzipBuffer.cpp")
        lib_dirs_dyna.append(os.path.join(femzip_path))
        libs_dyna = ['femunziplib_standard_dyna', 'ipp_zlibd', 'ippcoremt',
                     'ippdcmt', 'ippsmt', 'ifwin', 'ifconsol', 'ippvmmt', 'libmmd',
                     'libirc', 'svml_dispmd', 'msvcrt']
        compiler_args_dyna.append("/DQD_USE_FEMZIP")
    elif (platform.system() == "Linux") and os.path.isdir(femzip_path):
        srcs_dyna.append("qd/cae/dyna_cpp/dyna/FemzipBuffer.cpp")
        lib_dirs_dyna.append(femzip_path)
        libs_dyna = ['femunzip_dyna_standard', 'ipp_z', 'ippcore',
                     'ippdc', 'ipps', 'ifcore_pic', 'ifcoremt', 'imf',
                     'ipgo', 'irc', 'svml', 'ippcore_l', 'stdc++', 'dl']
        compiler_args_dyna.append("-DQD_USE_FEMZIP")
else:
    print("FEMZIP library %s not found. Compiling without femzip support." % femzip_path)

# CFLAGS linux
if (platform.system().lower() == "linux") or (platform.system().lower() == "linux2"):
    compiler_args_dyna.append("-std=c++14")
    compiler_args_dyna.append("-O3")
    compiler_args_dyna.append("-fPIC")
    if debugging_mode:
        compiler_args_dyna.append("-DQD_DEBUG")
    if measure_time:
        compiler_args_dyna.append("-DQD_MEASURE_TIME")
# CFLAGS Windows
else:
    compiler_args_dyna.append("/EHa")
    if debugging_mode:
        compiler_args_dyna.append("/DQD_DEBUG")
    if measure_time:
        compiler_args_dyna.append("/DQD_MEASURE_TIME")
dyna_extension = Extension("dyna_cpp", srcs_dyna,
                           extra_compile_args=compiler_args_dyna,
                           library_dirs=lib_dirs_dyna,
                           libraries=libs_dyna,
                           include_dirs=include_dirs_dyna,)

# (2) UNIT-TESTING


def my_test_suite():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('test', pattern='test_*.py')
    return test_suite


# (3) SETUP
setup(name='qd',
      version=_version,
      license='GNU GPL v3',
      description='QD-Engineering Python Library for CAE',
      author='C. Diez, D. Toewe',
      url='http://www.qd-eng.de',
      author_email='qd.eng.contact@gmail.com',
      packages=(['qd',
                 'qd.cae',
                   'qd.cae.beta',
                   'qd.cae.resources',
                   'qd.numerics',
                 ]),
      package_dir={'qd': 'qd',
                   'qd.cae': 'qd/cae',
                   'qd.cae.beta': 'qd/cae/beta',
                   'qd.cae.resources': 'qd/cae/resources',
                   'qd.numerics': 'qd/numerics',
                   },
      package_data={
          'qd.cae.resources': ['*.js', 'html.template'],
          'qd.cae.beta': ['meta_remote_control', 'meta_remote_control.exe']
      },
      ext_package='qd.cae',  # where to place c extensions
      ext_modules=[dyna_extension],
      install_requires=['numpy>=1.11', 'diversipy', 'pybind11>2.1.0'],
      keywords=['cae',
                  'simulation',
                  'engineering',
                  'ls-dyna',
                  'postprocessing',
                  'preprocessing'],
      classifiers=['Development Status :: 4 - Beta',
                   'Programming Language :: C++',
                   'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                   'Topic :: Scientific/Engineering',
                   'Intended Audience :: Science/Research',
                   'Topic :: Utilities',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6'],
      test_suite='setup.my_test_suite',)
