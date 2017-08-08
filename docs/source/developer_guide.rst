
Developer Guide
===============

This guide is meant for people, who want to extend this package.


Compilation
-----------

First step is to get the package to compile. Therefore visit the `compilation guide`_ for your respective OS.

.. _compilation guide: https://qd-cae.github.io/qd-cae-python/build/html/compilation_guide.html


Adding new Classes
------------------

**Coding guideline**

Even though I also don't stick rigidly to it, please follow the `google coding style`_. If you don't know it yet, then read it. It really benefits you as programmer in general. 

.. _google coding style: https://google.github.io/styleguide/cppguide.html

The code is written in modern C++14. This means:

  - use smart pointers for storage (no raw ones): `#include <memory>`
  - use standardized integers: `#include <cstdint>`
  - use static compiler checks

**Creating new files**

The first step is to create a new source file (.cpp) and new header file (.hpp). Don't forget the header guards, and also embed your code in the *namespace qd*.  Example case:

*my_class.hpp*

.. code-block:: cpp

    #ifndef MY_CLASS_HPP
    #define MY_CLASS_HPP

    // includes
    #include <cstdint>

    namepspace qd {

    class MyClass {
        int32_t some_int;

        MyClass(int32_t _some_int);
        int32_t get_id();
    };

    } // namespace qd

    #endif // MY_CLASS_HPP

*my_class.cpp*

.. code-block:: cpp

    #include <my_class.hpp>

    namepspace qd {

    MyClass::MyClass(int32_t _some_int)
        : some_int(_some_int) // always initialize variables!
    {
        // nothing here
    }

    int32_t MyClass::get_id()
    {
        return this->some_int;
    }

    } // namespace qd


**Adding files to the build process**

The new source file (not header) must be added to the `setup.py`. There is a list variable called `srcs_dyna`. Simply append your new source file.

Connecting C++ and Python
-------------------------

You should already be able to compile the package now with your new source file (simply try it). It is now time to connect your C++ class to Python. The wrapper library for it is `pybind11`_. First open the file, which compiles the API, called: **pybind_wrapper.cpp**. A class can be added quite easily:

.. _pybind11: https://github.com/pybind/pybind11

.. code-block:: cpp

    pybind11::class_<MyClass, std::shared_ptr<MyClass>> 
        my_class_py(m, "MyClass", my_class_description);

The variable `my_class_description` is a `const char*` and contains the documentation of the file (numpydoc style). Documentation strings are added to the file `docstrings.cpp`, but in the beginning simply ignore this. The `shared_ptr` tells `pybind11`_ to use a `shared_ptr` as holder type.

First we need to add the constructor of the class.

.. code-block:: cpp

    my_class_py.def(pybind11::init<int32_t>(),
                    "id"_a)

The first line tells pybind11 to make a constructor and the second one specifies, that the first argument has the name `id`. One may assign a default value here by the way. The literal ending `_a` is a string literal, telling pybind11, that we define the argument name here. Pybind11 handles the conversion of basic datatypes, such as `int` etc. automatically. It is getting more complicated, if a `std::vector` is needed as argument (there is a conversion routine in the utilities, which converts python containers to C++ containers).

Finally we also add the class function.

.. code-block:: cpp

    my_class_py.def("get_id",
                    &MyClass::get_id,
                    pybind11::return_value_policy::take_ownership)

The string `get_id` is the name of the function in python. The `pybind11::return_value_policy` explicitly specifies how to deal with the value in memory. For raw data, such as an int here, it does not really make any difference. Pointers though might need a different return policy.

Here we are, lets compile and run the code:

.. code-block:: python

    from qd.cae.dyna import MyClass
    mc = MyClass(3)
    mc.get_id()
    >>> 3


Internal Data Structure
-----------------------
**TODO**

.. uml::
    
    @startuml
    
    FEMFile --|> DB_Nodes
    FEMFile --|> DB_Elements
    FEMFile --|> DB_Parts
    D3plot  --|> FEMFile
    KeyFile --|> FEMFile
    @enduml


Documentation
-------------
**TODO**

