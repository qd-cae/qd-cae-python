CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET(QD_CAE_DYNA_SRC qd/cae/dyna_cpp/db/FEMFile.cpp
        qd/cae/dyna_cpp/db/DB_Elements.cpp
        qd/cae/dyna_cpp/db/DB_Nodes.cpp
        qd/cae/dyna_cpp/db/DB_Parts.cpp
        qd/cae/dyna_cpp/db/Element.cpp
        qd/cae/dyna_cpp/db/Node.cpp
        qd/cae/dyna_cpp/db/Part.cpp
        qd/cae/dyna_cpp/dyna/d3plot/D3plotBuffer.cpp
        qd/cae/dyna_cpp/dyna/d3plot/D3plot.cpp
        qd/cae/dyna_cpp/dyna/d3plot/RawD3plot.cpp
        qd/cae/dyna_cpp/dyna/d3plot/FemzipBuffer.cpp
        qd/cae/dyna_cpp/dyna/keyfile/KeyFile.cpp
        qd/cae/dyna_cpp/dyna/keyfile/Keyword.cpp
        qd/cae/dyna_cpp/dyna/keyfile/NodeKeyword.cpp
        qd/cae/dyna_cpp/dyna/keyfile/ElementKeyword.cpp
        qd/cae/dyna_cpp/dyna/keyfile/PartKeyword.cpp
        qd/cae/dyna_cpp/dyna/keyfile/IncludeKeyword.cpp
        qd/cae/dyna_cpp/dyna/keyfile/IncludePathKeyword.cpp
        qd/cae/dyna_cpp/utility/FileUtility.cpp
        qd/cae/dyna_cpp/utility/TextUtility.cpp
        qd/cae/dyna_cpp/parallel/WorkQueue.cpp)

# Bare c++ module
ADD_LIBRARY( QD_CAE_DYNA_LIB STATIC "${QD_CAE_DYNA_SRC}")
INCLUDE_DIRECTORIES(qd/cae)

# FEMZIP
SET(FEMZIP_LIB_DIR "${CMAKE_SOURCE_DIR}/libs/femzip/FEMZIP_8.68_dyna_NO_OMP_Windows_VS2012_MD_x64/x64")
#LINK_DIRECTORIES(FEMZIP_LIB_DIR)
SET(FEMZIP_LIBS "${FEMZIP_LIB_DIR}/femunziplib_standard_dyna.lib" 
                "${FEMZIP_LIB_DIR}/ipp_zlibd.lib" 
                "${FEMZIP_LIB_DIR}/ippcoremt.lib" 
                "${FEMZIP_LIB_DIR}/ippdcmt.lib" 
                "${FEMZIP_LIB_DIR}/ippsmt.lib" 
                "${FEMZIP_LIB_DIR}/ifwin.lib" 
                "${FEMZIP_LIB_DIR}/ifconsol.lib" 
                "${FEMZIP_LIB_DIR}/ippvmmt.lib" 
                "${FEMZIP_LIB_DIR}/libmmd.lib" 
                "${FEMZIP_LIB_DIR}/libirc.lib" 
                "${FEMZIP_LIB_DIR}/svml_dispmd.lib" 
                "msvcrt") 
ADD_DEFINITIONS(-DQD_USE_FEMZIP)
TARGET_LINK_LIBRARIES(QD_CAE_DYNA_LIB "${FEMZIP_LIBS}")

# OPENMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# TEST Executable
ADD_EXECUTABLE(test_app "${CMAKE_SOURCE_DIR}/test/test_app.cpp")
TARGET_LINK_LIBRARIES(test_app QD_CAE_DYNA_LIB "${FEMZIP_LIBS}")
