#!/usr/bin/env python

import os
import sys
import platform
import compileall
from setuptools import find_packages,setup,Extension

# ======= S E T T I N G S ======= #
debugging_mode = False
useFemzip = True
# =============================== #

# Version
_version = "0.1.3"

# py -> pyc
compileall.compile_dir('src/')

# (0) Compiler Stuff
# Check for MinGW usage
use_mingw = False
for ii in range(len(sys.argv)):
	if (sys.argv[ii] == "-c") and (sys.argv[ii+1] == "mingw32"):
		use_mingw=True
	if (sys.argv[ii] == "--compiler=mingw32"):
		use_mingw=True

# (1) Native Code Stuff
# (1.1) DYNA-POST toolbox
if not os.path.isdir("libs/boost_1_61_0"):
    raise Exception("You are missing the library: libs/boost_1_61_0.")
    #b2 --toolset=msvc-10.0 --build-type=complete architecture=x86 address-model=64 stage
compiler_args_dyna = []
include_dirs_dyna = ["libs/boost_1_61_0"]
lib_dirs_dyna = [] # ["libs/boost_1_61_0/lib64-msvc-9.0"]
libs_dyna  = [] # ["boost_python"]
srcs_dyna = ["src/qd/cae/dyna_cpp/python_api/wrapper.cpp",
    "src/qd/cae/dyna_cpp/db/FEMFile.cpp",
    "src/qd/cae/dyna_cpp/db/DB_Elements.cpp",
    "src/qd/cae/dyna_cpp/db/DB_Nodes.cpp",
    "src/qd/cae/dyna_cpp/db/DB_Parts.cpp",
    "src/qd/cae/dyna_cpp/db/Element.cpp",
    "src/qd/cae/dyna_cpp/db/Node.cpp",
    "src/qd/cae/dyna_cpp/db/Part.cpp",
    "src/qd/cae/dyna_cpp/dyna/D3plotBuffer.cpp",
    "src/qd/cae/dyna_cpp/dyna/D3plot.cpp",
    "src/qd/cae/dyna_cpp/dyna/KeyFile.cpp",
    "src/qd/cae/dyna_cpp/utility/FileUtility.cpp",
    "src/qd/cae/dyna_cpp/utility/TextUtility.cpp",
    "src/qd/cae/dyna_cpp/utility/MathUtility.cpp"]
# FEMZIP usage? Libraries present?
# You need to download the femzip libraries yourself from SIDACT GmbH
# If you have questions, write a mail.
if useFemzip:
    if (platform.system() == "Windows") and os.path.isdir('libs/Windows_VS2010_MT/x64'):
        srcs_dyna.append("src/qd/cae/dyna_cpp/dyna/FemzipBuffer.cpp")
        lib_dirs_dyna.append('libs/Windows_VS2010_MT/x64')
        libs_dyna = ['femunziplib_standard_dyna','ipp_zlib','ippcoremt',
            'ippdcmt','ippsmt','ifwin','ifconsol','ippvmmt','libmmt',
            'libirc','svml_dispmt','msvcrt']
        compiler_args_dyna.append("/DQD_USE_FEMZIP")
    elif (platform.system() == "Linux") and os.path.isdir('libs/Linux/64Bit'):
        srcs_dyna.append("src/qd/cae/dyna_cpp/dyna/FemzipBuffer.cpp")
        lib_dirs_dyna.append('libs/Linux/64Bit')
        libs_dyna = ['femunzip_dyna_standard','ipp_z','ippcore',
            'ippdc','ipps','ifcore_pic','ifcoremt','imf',
            'ipgo','irc','svml','ippcore_l','stdc++','dl']
        compiler_args_dyna.append("-DQD_USE_FEMZIP")
# CFLAGS linux
if (platform == "linux") or (platform == "linux2") or use_mingw:
	#compiler_args_dyna.append("-std=c++11")
	compiler_args_dyna.append("-O3")
	if not use_mingw:
		compiler_args_dyna.append("-fPIC")
	if debugging_mode:
		compiler_args_dyna.append("-DQD_DEBUG")
# CFLAGS Windows
else:
	if debugging_mode:
		compiler_args_dyna.append("/DQD_DEBUG")
dyna_extension = Extension("dyna_post", srcs_dyna, extra_compile_args = compiler_args_dyna,
									  library_dirs=lib_dirs_dyna,
									  libraries=libs_dyna,
									  include_dirs=include_dirs_dyna,)


# (2) setup
setup(name = 'qd',
		version = _version,
		license = 'GNU GPL v3',
		description = 'QD-Engineering python library',
		author = 'C. Diez, N. Klinke',
        url    = 'www.qd-eng.de',
        author_email = 'constantin.diez@gmail.com',
		packages=(['qd',
                 'qd.cae',
                 'qd.numerics']),
		#packages=find_packages(),
		package_dir={'qd'    : 'src/qd',
                     'qd.cae' : 'src/qd/cae',
                     'qd.numerics' : 'src/qd/numerics'},
        ext_package='qd.cae', # where to place c extensions
        ext_modules=[dyna_extension],
		#install_requires=['openpyxl','matplotlib','numpy','codie'],
		keywords=['cae','engineering','ls-dyna','postprocessing','preprocessing'],
        classifiers=['Development Status :: 3 - Alpha',
                     'Programming Language :: C++',
                     'Topic :: Scientific/Engineering',
                     'Intended Audience :: Science/Research',
                     'Topic :: Utilities',
                     'Topic :: Engineering',
                     'Topic :: CAE',
                     'Topic :: FEM',
                     'Programming Language :: Python :: 2.7'],)
