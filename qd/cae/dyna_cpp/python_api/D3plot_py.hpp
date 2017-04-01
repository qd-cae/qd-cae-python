
#ifndef D3PLOT_PY
#define D3PLOT_PY

// forward declarations
class D3plot;

// includes
#include <Python.h>
#include "FEMFile_py.hpp"

// namespaces
//using namespace std;

extern "C" {

  /* OBJECT */
  typedef struct {
      QD_FEMFile femfile; // Base
      // Type-specific fields go here.
      D3plot* d3plot;
  } QD_D3plot;


  /* DEALLOC */
  static void
  QD_D3plot_dealloc(QD_D3plot* self);

  /* NEW */
  static PyObject *
  QD_D3plot_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

  /* INIT */
  static int
  QD_D3plot_init(QD_D3plot *self, PyObject *args, PyObject *kwds);

  /* FUNCTION get_timesteps */
  static PyObject *
  QD_D3plot_get_timesteps(QD_D3plot* self);

  const char* get_timesteps_docs = "\
get_timesteps()\n\
\n\
Returns\n\
-------\n\
timesteps : np.ndarray\n\
    state timesteps of the D3plot\n\
\n\
Examples\n\
--------\n\
    >>> d3plot = D3plot(\"path/to/d3plot\")\n\
    >>> time = d3plot.get_timesteps()\n\
";

  /* FUNCTION read_states */
  static PyObject *
  QD_D3plot_read_states(QD_D3plot* self, PyObject* args);

  const char* read_states_docs = "\
read_states(vars)\n\
\n\
Parameters\n\
----------\n\
vars : str or list(str)\n\
    variable or list of variables to read\n\
\n\
Returns\n\
-------\n\
timesteps : np.ndarray\n\
    state timesteps of the D3plot\n\
Notes\n\
-----\n\
    Read a variable from the state files. If this is not done, the nodes\n\
    and elements return empty vectors when requesting a result. The\n\
    variables available are:\n\
    \n\
    * disp (displacement)\n\
    * vel (velocity)\n\
    * accel (acceleration)\n\
    * strain [(optional) mode]\n\
    * stress [(optional) mode]\n\
    * stress_mises [(optional) mode]\n\
    * plastic_strain [(optional) mode]\n\
    * history [id1] [id2] [shell or solid] [(optional) mode]\n\
    \n\
    There is an optional mode for the element results. Since shell elements\n\
    have multiple layers of results, the optional mode determines the\n\
    treatment of these layers:\n\
    \n\
    * inner\n\
    * mid (middle)\n\
    * outer\n\
    * mean\n\
    * max\n\
    * min\n\
\n\
Examples\n\
--------\n\
    >>> d3plot = D3plot(\"path/to/d3plot\") # just geometry\n\
    >>> node = d3plot.get_nodeByID(1)\n\
    >>> len( node.get_disp() ) # no disps loaded\n\
    0\n\
    >>> # Read displacements\n\
    >>> d3plot.read_states(\"disp\")\n\
    >>> len( node.get_disp() ) # here they are\n\
    31\n\
    >>> # multi-loading, already loaded will be skipped\n\
    >>> d3plot.read_states([\"disp\",\"vel\",\"stress_mises max\",\"shell history 1 mean\"])\n\
    >>> # most efficient way, load the results directly when opening\n\
    >>> D3plot(\"path/to/d3plot\", read_states=[\"disp\",\"vel\",\"plastic_strain max\"])\n\
";

  /* FUNCTION info */
  static PyObject *
  QD_D3plot_info(QD_D3plot* self);

  const char* info_docs = "\
info()\n\
\n\
Notes\n\
-----\n\
    Get information about the header data of the D3plot.\n\
\n\
Examples\n\
--------\n\
    >>> d3plot = D3plot(\"path/to/d3plot\")\n\
    >>> d3plot.info()\n\
";

  /* FUNCTION clear */
  static PyObject *
  QD_D3plot_clear(QD_D3plot* self, PyObject* args);

  const char* clear_docs = "\
clear(vars)\n\
\n\
Parameters\n\
----------\n\
vars : str or list(str)\n\
    variable or list of variables to delete\n\
\n\
Notes\n\
-----\n\
    This function may be used if one wants to clear certain state data\n\
    from the memory. Valid variable names are:\n\
    \n\
    * disp\n\
    * vel\n\
    * accel\n\
    * strain\n\
    * stress\n\
    * stress_mises\n\
    * plastic_strain\n\
    * history [(optional) shell or solid]\n\
    \n\
    The specification of shell or solid for history is optional. Deletes\n\
    all history variables if none given.\n\
\n\
Examples\n\
--------\n\
    >>> d3plot = D3plot(\"path/to/d3plot\", read_states=\"strain inner\")\n\
    >>> elem = d3plot.get_elementByID(\"shell\",1)\n\
    >>> len(elem.get_strain())\n\
    34\n\
    >>> # clear specific field\n\
    >>> d3plot.clear(\"strain\")\n\
    >>> len(elem.get_strain())\n\
    0\n\
    >>> # reread some data\n\
    >>> d3plot.read_states(\"strain outer\")\n\
    >>> len(elem.get_strain())\n\
    34\n\
    >>> d3plot.clear() # clear all\n\
";

  /* METHOD TABLE */
  static PyMethodDef QD_D3plot_methods[] = {
    {"get_timesteps", (PyCFunction) QD_D3plot_get_timesteps, METH_NOARGS, get_timesteps_docs},
    {"read_states", (PyCFunction) QD_D3plot_read_states, METH_VARARGS, read_states_docs},
    {"info", (PyCFunction) QD_D3plot_info, METH_NOARGS, info_docs},
    {"clear", (PyCFunction) QD_D3plot_clear, METH_VARARGS, clear_docs},
	  {NULL}  /* Sentinel */
  };


  /* TYPE ... whatever */
  static PyTypeObject QD_D3plot_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "QD_D3plot",               /* tp_name */
    sizeof(QD_D3plot),         /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)QD_D3plot_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT |
    Py_TPFLAGS_BASETYPE,        /* tp_flags */
    "QD_D3plot",                  /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    QD_D3plot_methods,         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    &QD_FEMFile_Type,          /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc) QD_D3plot_init, /* tp_init */
    0,                         /* tp_alloc */
    QD_D3plot_new,             /* tp_new */
  };

}

#endif
