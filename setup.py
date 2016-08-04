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
_version = "0.1.0"

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
# (1.1) DYNA toolbox
compiler_args_dyna = []
include_dirs_dyna = []
lib_dirs_dyna = []
libs_dyna = []
srcs_dyna = ["src/qd/cae/dyna/python_api/wrapper.cpp",
    "src/qd/cae/dyna/db/DB_Elements.cpp",
    "src/qd/cae/dyna/db/DB_Nodes.cpp",
    "src/qd/cae/dyna/db/DB_Parts.cpp",
    "src/qd/cae/dyna/db/Element.cpp",
    "src/qd/cae/dyna/db/Node.cpp",
    "src/qd/cae/dyna/db/Part.cpp",
    "src/qd/cae/dyna/dyna/D3plotBuffer.cpp",
    "src/qd/cae/dyna/dyna/d3plot.cpp",
    "src/qd/cae/dyna/utility/FileUtility.cpp",
    "src/qd/cae/dyna/utility/IOUtility.cpp",
    "src/qd/cae/dyna/utility/TextUtility.cpp",
    "src/qd/cae/dyna/utility/MathUtility.cpp"]

# FEMZIP usage? Libraries present?
if useFemzip:
    if (platform.system() == "Windows") and os.path.isdir('libs/Windows_VS2010_MT/x64'):
        srcs_dyna.append("src/qd/cae/dyna/dyna/FemzipBuffer.cpp")
        lib_dirs_dyna = ['libs/Windows_VS2010_MT/x64']
        libs_dyna = ['femunziplib_standard_dyna','ipp_zlib','ippcoremt',
            'ippdcmt','ippsmt','ifwin','ifconsol','ippvmmt','libmmt',
            'libirc','svml_dispmt','msvcrt']
        compiler_args_dyna.append("/DCD_USE_FEMZIP")
    elif (platform.system() == "Linux") and os.path.isdir('libs/Linux/64Bit'):
        srcs_dyna.append("src/qd/cae/dyna/dyna/FemzipBuffer.cpp")
        lib_dirs_dyna = ['libs/Linux/64Bit']
        libs_dyna = ['femunzip_dyna_standard','ipp_z','ippcore',
            'ippdc','ipps','ifcore_pic','ifcoremt','imf',
            'ipgo','irc','svml','ippcore_l','stdc++','dl']
        compiler_args_dyna.append("-DCD_USE_FEMZIP")
# CFLAGS linux
if (platform == "linux") or (platform == "linux2") or use_mingw:
	compiler_args_dyna.append("-std=c++11")
	compiler_args_dyna.append("-O3")
	if not use_mingw:
		compiler_args_dyna.append("-fPIC")
	if debugging_mode:
		compiler_args_dyna.append("-DQD_DEBUG")
# CFLAGS Windows
else:
	if debugging_mode:
		compiler_args_dyna.append("/DQD_DEBUG")
dyna_extension = Extension("dyna", srcs_dyna, extra_compile_args = compiler_args_dyna,
									  library_dirs=lib_dirs_dyna,
									  libraries=libs_dyna,
									  include_dirs=include_dirs_dyna,)

# (2) setup
setup(name = 'qd',
		version = _version,
		license = 'GNU GPL v3',
		description = 'qd-engineering python library',
		author = 'C. Diez, N. Klinke',
        url    = 'www.qd-eng.de',
        author_email = 'gmjason@gmx.de',
		packages=(['qd',
                 'qd.cae',
                 'qd.cae.dyna',
                 'qd.numerics']),
		#packages=find_packages(),
		package_dir={'qd'    : 'src/qd',
                     'qd.cae' : 'src/qd/cae',
                     'qd.numerics' : 'src/qd/numerics'},
        ext_package='qd.cae', # where to place c extensions
        ext_modules=[dyna_extension],
		#install_requires=['openpyxl','matplotlib','numpy','codie'],
		keywords=['cae','engineering','ls-dyna'],
        classifiers=['Development Status :: 4 - Beta',
                     'Topic :: Utilities',
                     'Topic :: Engineering',
                     'Topic :: CAE',
                     'Topic :: FEM',
                     'Programming Language :: Python :: 2.7'],)
      