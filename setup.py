#!/usr/bin/env python

import os
import sys
import glob
import platform
import numpy as np
import unittest
from setuptools import setup, Extension

try:
    import pybind11
except ImportError:
    import pip
    pip.main(['install', 'pybind11==2.1.1'])
    import pybind11

# ======= S E T T I N G S ======= #
femzip_path_windows = "libs/femzip/FEMZIP_8.68_dyna_NO_OMP_Windows_VS2012_MD_x64/x64"  # optional
femzip_path_linux = "libs/femzip/Linux64/64Bit/"  # optional
# ====== D E V E L O P E R ====== #
debugging_mode = False
measure_time = False
use_openmp = False  # buggy
version = "0.8.9"
# =============================== #
is_windows = (platform.system() == "Windows")
is_linux = (platform.system() in ["Linux", "Darwin"])
# =============================== #


def setup_dyna_cpp():
    '''Basic setup of the LS-DYNA C-Extension

    Returns
    -------
    srcs : list of str
    include_dirs : list of str
    compiler_args : list of str
    '''

    include_dirs = ["qd/cae",
                    np.get_include(),
                    pybind11.get_include()]
    srcs = [
        "qd/cae/dyna_cpp/python_api/pybind_wrapper.cpp",
        # "qd/cae/dyna_cpp/python_api/docstrings.cpp",
        "qd/cae/dyna_cpp/db/FEMFile.cpp",
        "qd/cae/dyna_cpp/db/DB_Elements.cpp",
        "qd/cae/dyna_cpp/db/DB_Nodes.cpp",
        "qd/cae/dyna_cpp/db/DB_Parts.cpp",
        "qd/cae/dyna_cpp/db/Element.cpp",
        "qd/cae/dyna_cpp/db/Node.cpp",
        "qd/cae/dyna_cpp/db/Part.cpp",
        "qd/cae/dyna_cpp/dyna/d3plot/D3plotBuffer.cpp",
        "qd/cae/dyna_cpp/dyna/d3plot/D3plot.cpp",
        "qd/cae/dyna_cpp/dyna/d3plot/RawD3plot.cpp",
        "qd/cae/dyna_cpp/dyna/keyfile/KeyFile.cpp",
        "qd/cae/dyna_cpp/dyna/keyfile/Keyword.cpp",
        "qd/cae/dyna_cpp/dyna/keyfile/NodeKeyword.cpp",
        "qd/cae/dyna_cpp/dyna/keyfile/ElementKeyword.cpp",
        "qd/cae/dyna_cpp/dyna/keyfile/PartKeyword.cpp",
        "qd/cae/dyna_cpp/dyna/keyfile/IncludeKeyword.cpp",
        "qd/cae/dyna_cpp/dyna/keyfile/IncludePathKeyword.cpp",
        "qd/cae/dyna_cpp/utility/FileUtility.cpp",
        "qd/cae/dyna_cpp/utility/TextUtility.cpp",
        # "qd/cae/dyna_cpp/parallel/WorkQueue.cpp",
    ]

    extra_link_args = []

    libs_dyna = []

    # linux compiler args
    if is_linux:
        compiler_args = ["-std=c++11",
                         "-fPIC",
                         "-D_GLIBCXX_USE_CXX11_ABI=0",  # ABI compatability
                         "-DQD_VERSION=\"" + version + "\""]

        libs_dyna = ["stdc++"]

        if debugging_mode:
            compiler_args.append("-DQD_DEBUG")
            compiler_args.append("-O0")
        else:
            compiler_args.append("-O3")
        if measure_time:
            compiler_args.append("-DQD_MEASURE_TIME")
        if use_openmp:
            compiler_args.append("-fopenmp")
            extra_link_args.append("-lgomp")
        else:
            compiler_args.append("-Wno-unknown-pragmas")

    # windowscompiler args
    elif is_windows:
        compiler_args = ["/EHa",
                         "/DQD_VERSION=\\\"" + version + "\\\""]
        if debugging_mode:
            compiler_args.append("/DQD_DEBUG")
        if measure_time:
            compiler_args.append("/DQD_MEASURE_TIME")
        if use_openmp:
            compiler_args.append("/openmp")

    else:
        raise RuntimeError("Could not determine os (windows or linux)")

    return srcs, include_dirs, compiler_args, extra_link_args, libs_dyna


def setup_dyna_cpp_binout(srcs, compiler_args):

    srcs = ["qd/cae/dyna_cpp/dyna/binout/Binout.cpp",
            "qd/cae/dyna_cpp/dyna/binout/lsda/lsda.c",
            "qd/cae/dyna_cpp/dyna/binout/lsda/btree.c",
            "qd/cae/dyna_cpp/dyna/binout/lsda/lsdatable.c",
            "qd/cae/dyna_cpp/dyna/binout/lsda/lsdatypes.c",
            "qd/cae/dyna_cpp/dyna/binout/lsda/trans.c",
            "qd/cae/dyna_cpp/dyna/binout/lsda/lsdaf2c.c"] + srcs

    if is_linux:
        compiler_args += ["-fpermissive", "-DQD_USE_C_BINOUT"]
    elif is_windows:
        compiler_args += ["/DQD_USE_C_BINOUT"]

    return srcs, compiler_args


def setup_dyna_cpp_femzip(srcs, lib_dirs, libs, compiler_args):
    ''' Checks for femzip libraries
    '''

    # Uses FEMZIP, iff link librares are present
    # You need to download the femzip libraries yourself from SIDACT GmbH
    # www.sidact.de
    # If you have questions, write a mail.

    # windows
    if is_windows and os.path.isdir(os.path.join(femzip_path_windows)):

        srcs.append("qd/cae/dyna_cpp/dyna/d3plot/FemzipBuffer.cpp")
        lib_dirs.append(os.path.join(femzip_path_windows))
        libs += ['femunziplib_standard_dyna', 'ipp_zlibd', 'ippcoremt',
                 'ippdcmt', 'ippsmt', 'ifwin', 'ifconsol', 'ippvmmt', 'libmmd',
                 'libirc', 'svml_dispmd', 'msvcrt']
        compiler_args.append("/DQD_USE_FEMZIP")

    # linux
    elif is_linux and os.path.isdir(femzip_path_linux):

        srcs.append("qd/cae/dyna_cpp/dyna/d3plot/FemzipBuffer.cpp")
        lib_dirs.append(femzip_path_linux)
        libs += ['femunzip_dyna_standard', 'ipp_z', 'ippcore',
                 'ippdc', 'ipps', 'ifcore_pic', 'ifcoremt', 'imf',
                 'ipgo', 'irc', 'svml', 'ippcore_l', 'stdc++', 'dl']
        compiler_args.append("-DQD_USE_FEMZIP")
    else:
        print("FEMZIP library not found. Compiling without FEMZIP.")

    return srcs, lib_dirs, libs, compiler_args


def setup_dyna_cpp_hdf5(srcs, compiler_args, libs, lib_dirs, include_dirs):
    ''' Sets up the hdf5 compilation
    '''

    srcs.append("qd/cae/dyna_cpp/utility/HDF5_Utility.cpp")

    if is_linux:
        compiler_args.append("-DQD_USE_HDF5")
        raise RuntimeError("Linux HDF5 compilation missing.")

    if is_windows:

        import ntpath
        files = glob.glob("libs/hdf5/windows/lib/lib*.lib")
        files = [ntpath.basename(entry).replace(".lib", "") for entry in files]
        libs += files

        lib_dirs.append("libs/hdf5/windows/lib")
        include_dirs.append("libs/hdf5/windows/include")
        compiler_args.append("/DQD_USE_HDF5")

    return srcs, compiler_args, libs, lib_dirs, include_dirs


def use_vtk_if_possible(depency_packages):

    # do nothing if missing
    if not os.path.isdir('./libs/VTK'):
        return []

    # packages
    depency_packages = ['autobahn', 'constantly', 'hyperlink',
                        'incremental', 'Twisted', 'txaio', 'wslink', 'zope.interface']

    ''' put in top of init
    import os
    import sys

    # add runtime libraries to environment
    dir_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), "shared")
    if dir_path not in sys.path:
        sys.path.insert(0, dir_path)
    '''

    return depency_packages


def my_test_suite():
    ''' Sets up the testing
    '''
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('test', pattern='test_*.py')
    return test_suite


if __name__ == "__main__":

    # setup basic extension
    lib_dirs_dyna = []
    srcs_dyna, include_dirs_dyna, compiler_args_dyna, extra_link_args, libs_dyna = setup_dyna_cpp()

    # compile binout
    # srcs_dyna, compiler_args_dyna = setup_dyna_cpp_binout(
    #     srcs_dyna, compiler_args_dyna)

    # setup hdf5
    # (MUST be before femzip, due to linking)
    '''
    srcs_dyna, \
    compiler_args_dyna, \
    libs_dyna, \
    lib_dirs_dyna, \
    include_dirs_dyna = setup_dyna_cpp_hdf5(srcs_dyna,
                                            compiler_args_dyna,
                                            libs_dyna,
                                            lib_dirs_dyna,
                                            include_dirs_dyna)
    '''

    # setup femzip (if possible)
    srcs_dyna, lib_dirs_dyna, libs_dyna, compiler_args_dyna = setup_dyna_cpp_femzip(
        srcs_dyna,
        lib_dirs_dyna,
        libs_dyna,
        compiler_args_dyna)

    # setup extension
    dyna_extension = Extension("dyna_cpp", srcs_dyna,
                               extra_compile_args=compiler_args_dyna,
                               extra_objects=[],
                               extra_link_args=extra_link_args,
                               library_dirs=lib_dirs_dyna,
                               libraries=libs_dyna,
                               include_dirs=include_dirs_dyna,)

    # (3) SETUP
    setup(name='qd',
          version=version,
          license='BSD3',
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
              'qd.cae.beta': ['meta_remote_control', 'meta_remote_control.exe', 'msvcr71.dll']
          },
          ext_package='qd.cae',  # where to place c extensions
          ext_modules=[dyna_extension],
          install_requires=['numpy>=1.8',
                            'diversipy', 'pybind11==2.1.1', 'h5py'],
          keywords=['cae',
                      'simulation',
                      'engineering',
                      'ls-dyna',
                      'postprocessing',
                      'preprocessing',
                      'keyfile',
                      'binout'],
          classifiers=['Development Status :: 4 - Beta',
                       'Programming Language :: C++',
                       'Topic :: Scientific/Engineering',
                       'Intended Audience :: Science/Research',
                       'Topic :: Utilities',
                       'Operating System :: Microsoft :: Windows',
                       'Operating System :: POSIX :: Linux',
                       'Programming Language :: Python :: 3.5',
                       'Programming Language :: Python :: 3.6',
                       'Programming Language :: Python :: 3.7'],
          test_suite='setup.my_test_suite',)
