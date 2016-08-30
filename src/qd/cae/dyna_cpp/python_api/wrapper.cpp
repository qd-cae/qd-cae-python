
#include <Python.h>
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
#include <set>
#include "../utility/TextUtility.hpp"
#include "../dyna/KeyFile.hpp"
#include "../dyna/D3plot.hpp"
#include "../db/DB_Elements.hpp"
#include "../db/DB_Nodes.hpp"
#include "../db/DB_Parts.hpp"
#include "../db/Node.hpp"
#include "../db/Part.hpp"
#include "../db/Element.hpp"

using namespace std;

extern "C" {

  /* convert_obj_to_int: cast obj to long with checks */
  static int
  convert_obj_to_int(PyObject* item){

    if(!PyInt_Check(item)){
          PyErr_SetString(PyExc_SyntaxError, "Error, argument list entry is not an integer.");
          throw(-1);
    }

    long nodeID_long = PyLong_AsLong(item);

    // Overflow cast check
    if((long) std::numeric_limits<int>::max() < nodeID_long){
      PyErr_SetString(PyExc_SyntaxError, "Integer overflow error.");
      throw(-1);
    } else if ((long) std::numeric_limits<int>::min() > nodeID_long){
      PyErr_SetString(PyExc_SyntaxError, "Integer underflow error.");
      throw(-1);
    }

    return (int) PyLong_AsLong(item);

  }

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
    {"test_codie",  test_codie, METH_VARARGS,"Execute a shell command."},
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
    /*
    if (import_cd_cae() < 0)
      return NULL;
    */
    if (m == NULL)
      // PY3 return NULL;
      return;

    Py_INCREF(&QD_D3plot_Type);
    PyModule_AddObject(m, "QD_D3plot", (PyObject *)&QD_D3plot_Type);

    Py_INCREF(&QD_KeyFile_Type);
    PyModule_AddObject(m, "QD_KeyFile", (PyObject *)&QD_KeyFile_Type);

    Py_INCREF(&QD_Node_Type);
    PyModule_AddObject(m, "QD_Node", (PyObject *)&QD_Node_Type);

    Py_INCREF(&QD_Element_Type);
    PyModule_AddObject(m, "QD_Element", (PyObject *)&QD_Element_Type);

    Py_INCREF(&QD_Element_Type);
    PyModule_AddObject(m, "QD_Part", (PyObject *)&QD_Part_Type);

    // PY3 return m;
  }

}
