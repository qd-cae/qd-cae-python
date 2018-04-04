CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

# Bare c++ module
ADD_LIBRARY( qd_cae_dyna SHARED qd/cae/dyna_cpp/db/FEMFile.cpp
        qd/cae/dyna_cpp/db/DB_Elements.cpp
        qd/cae/dyna_cpp/db/DB_Nodes.cpp
        qd/cae/dyna_cpp/db/DB_Parts.cpp
        qd/cae/dyna_cpp/db/Element.cpp
        qd/cae/dyna_cpp/db/Node.cpp
        qd/cae/dyna_cpp/db/Part.cpp
        qd/cae/dyna_cpp/dyna/D3plotBuffer.cpp
        qd/cae/dyna_cpp/dyna/D3plot.cpp
        qd/cae/dyna_cpp/dyna/RawD3plot.cpp
        qd/cae/dyna_cpp/dyna/KeyFile.cpp
        qd/cae/dyna_cpp/dyna/Keyword.cpp
        qd/cae/dyna_cpp/dyna/NodeKeyword.cpp
        qd/cae/dyna_cpp/dyna/ElementKeyword.cpp
        qd/cae/dyna_cpp/dyna/PartKeyword.cpp
        qd/cae/dyna_cpp/dyna/IncludeKeyword.cpp
        qd/cae/dyna_cpp/dyna/IncludePathKeyword.cpp
        qd/cae/dyna_cpp/utility/FileUtility.cpp
        qd/cae/dyna_cpp/utility/TextUtility.cpp)

INCLUDE_DIRECTORIES(qd/cae)
        
#TARGET_LINK_LIBRARIES(qd_cae_dyna)

# compiler stuff
IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_DEFINITIONS("-Wall")
ENDIF()