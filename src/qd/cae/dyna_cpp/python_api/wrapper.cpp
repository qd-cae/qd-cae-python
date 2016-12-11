
#include <Python.h>
#include <numpy/arrayobject.h>
//#include "FEMFile_py.cpp"
#include "D3plot_py.hpp"
#include "Node_py.hpp"
#include "Element_py.hpp"
#include "Part_py.hpp"
#include "KeyFile_py.hpp"
#include <limits>
#include <string.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <algorithm>
#include <set>
#include "../utility/TextUtility.hpp"
#include "../utility/QD_Time.hpp"
#include "../dyna/KeyFile.hpp"
#include "../dyna/D3plot.hpp"
#include "../db/DB_Elements.hpp"
#include "../db/DB_Nodes.hpp"
#include "../db/DB_Parts.hpp"
#include "../db/Node.hpp"
#include "../db/Part.hpp"
#include "../db/Element.hpp"

using namespace std;

/** Convert a c++ 2D vector into a numpy array
 *
 * @param const vector< vector<T> >& vec : 2D vector data
 * @return PyArrayObject* array : converted numpy array
 *
 * Transforms an arbitrary 2D C++ vector into a numpy array. Throws in case of
 * unregular shape. The array may contain empty columns or something else, as
 * long as it's shape is square.
 *
 * Warning this routine makes a copy of the memory!
 */
template<typename T>
static PyArrayObject* vector_to_nparray(const vector< vector<T> >& vec, int type_num = PyArray_FLOAT){

   // rows not empty
   if( !vec.empty() ){

      // column not empty
      if( !vec[0].empty() ){

        size_t nRows = vec.size();
        size_t nCols = vec[0].size();
        npy_intp dims[2] = {nRows, nCols};
        PyArrayObject* vec_array = (PyArrayObject *) PyArray_SimpleNew(2, dims, type_num);

        T *vec_array_pointer = (T*) PyArray_DATA(vec_array);

        // copy vector line by line ... maybe could be done at one
        for (size_t iRow=0; iRow < vec.size(); ++iRow){

          if( vec[iRow].size() != nCols){
             Py_DECREF(vec_array); // delete
             throw(string("Can not convert vector<vector<T>> to np.array, since c++ matrix shape is not uniform."));
          }

          copy(vec[iRow].begin(),vec[iRow].end(),vec_array_pointer+iRow*nCols);
        }

        return vec_array;

     // Empty columns
     } else {
        npy_intp dims[2] = {vec.size(), 0};
        return (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_FLOAT, 0);
     }


   // no data at all
   } else {
      npy_intp dims[2] = {0, 0};
      return (PyArrayObject*) PyArray_ZEROS(2, dims, PyArray_FLOAT, 0);
   }

}


/** Convert a c++ vector into a numpy array
 *
 * @param const vector<T>& vec : 1D vector data
 * @return PyArrayObject* array : converted numpy array
 *
 * Transforms an arbitrary C++ vector into a numpy array. Throws in case of
 * unregular shape. The array may contain empty columns or something else, as
 * long as it's shape is square.
 *
 * Warning this routine makes a copy of the memory!
 */
template<typename T>
static PyArrayObject* vector_to_nparray(const vector<T>& vec, int type_num = PyArray_FLOAT){

   // rows not empty
   if( !vec.empty() ){

       size_t nRows = vec.size();
       npy_intp dims[1] = {nRows};

       PyArrayObject* vec_array = (PyArrayObject *) PyArray_SimpleNew(1, dims, type_num);
       T *vec_array_pointer = (T*) PyArray_DATA(vec_array);

       copy(vec.begin(),vec.end(),vec_array_pointer);
       return vec_array;

   // no data at all
   } else {
      npy_intp dims[1] = {0};
      return (PyArrayObject*) PyArray_ZEROS(1, dims, PyArray_FLOAT, 0);
   }

}


/** convert_obj_to_int: cast python object to long with checks
 *
 * @param PyObject* item : python object to convert
 * @return int ret : converted long
 *
 * The checks usually are done in python > 3 automatically but in python27
 * one has to deal with dirty stuff oneself.
 */
static int
convert_obj_to_int(PyObject* item){

  if(!PyInt_Check(item)){
        PyErr_SetString(PyExc_SyntaxError, "Error, argument list entry is not an integer.");
        throw(-1);
  }

  long nodeID_long = PyLong_AsLong(item);

  // Overflow cast check
  if((long) std::numeric_limits<int>::max < nodeID_long){
    throw(string("Integer overflow error."));
    //PyErr_SetString(PyExc_SyntaxError, "Integer overflow error.");
  } else if ((long) std::numeric_limits<int>::min > nodeID_long){
    throw(string("Integer underflow error."));
  }

  return (int) PyLong_AsLong(item);

}



extern "C" {


  // Include all the source implementations code
  //
  // This is a little weird, since they are just copied in here,
  // but who cares as long as it works fine.
  #include "FEMFile_py.cpp"
  #include "KeyFile_py.cpp"
  #include "D3plot_py.cpp"
  #include "Node_py.cpp"
  #include "Element_py.cpp"
  #include "Part_py.cpp"

  /*******************************************************/
  /*                                                     */
  /*                   Q D _ M O D U L E                 */
  /*                                                     */
  /*******************************************************/

  static PyObject *
  test_codie(PyObject *self, PyObject *args)
  {
   return Py_None;
  }

  /* MODULE codie function table */
  static PyMethodDef QDMethods[] = {
    {"test_codie",  test_codie, METH_VARARGS,"Used for debugging sometimes."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
  };

  /* MODULE codie */
  /* PYTHON 3 STUFF ?!?!?
  static PyModuleDef codie_module = {
    PyModuleDef_HEAD_INIT,
    "codie",
    "Codie Python cae module.",
    -1,
	CodieMethods
    //NULL, NULL, NULL, NULL, NULL
  };
  */

  /* MODULE INIT codie */
  /* PY3
  PyMODINIT_FUNC
  PyInit_codie(void)
  */
  void
  initdyna_cpp(void)
  {
    PyObject* m;

    // Constructor
    if (PyType_Ready(&QD_D3plot_Type) < 0)
      // PY3 return NULL;
      return;

    if (PyType_Ready(&QD_Node_Type) < 0)
      // PY3 return NULL;
      return;

    if (PyType_Ready(&QD_Element_Type) < 0)
      // PY3 return NULL;
      return;

    if (PyType_Ready(&QD_Part_Type) < 0)
      // PY3 return NULL;
      return;

   if (PyType_Ready(&QD_KeyFile_Type) < 0)
      // PY3 return NULL;
      return;

    // Init Module
    // Python 2.7
    m = Py_InitModule3("dyna_cpp", QDMethods,
                       "qd cae routines for LS-DYNA.");
    // PY3 m = PyModule_Create(&codie_module);

    import_array();

    if (m == NULL) 
      // PY3 return NULL;
      return;

    Py_INCREF(&QD_D3plot_Type);
    PyModule_AddObject(m, "QD_D3plot", (PyObject *)&QD_D3plot_Type);

    Py_INCREF(&QD_KeyFile_Type);
    PyModule_AddObject(m, "KeyFile", (PyObject *)&QD_KeyFile_Type);

    Py_INCREF(&QD_Node_Type);
    PyModule_AddObject(m, "Node", (PyObject *)&QD_Node_Type);

    Py_INCREF(&QD_Element_Type);
    PyModule_AddObject(m, "Element", (PyObject *)&QD_Element_Type);

    Py_INCREF(&QD_Element_Type);
    PyModule_AddObject(m, "Part", (PyObject *)&QD_Part_Type);

    // PY3 return m;
  }

}
