
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

  /* FUNCTION read_states */
  static PyObject *
  QD_D3plot_read_states(QD_D3plot* self, PyObject* args);

  /* FUNCTION info */
  static PyObject *
  QD_D3plot_info(QD_D3plot* self);


  /* METHOD TABLE */
  static PyMethodDef QD_D3plot_methods[] = {
    {"get_timesteps", (PyCFunction) QD_D3plot_get_timesteps, METH_NOARGS, "Get the timesteps in the d3plot."},
    {"read_states", (PyCFunction) QD_D3plot_read_states, METH_VARARGS, "Read a state variable."},
    {"info", (PyCFunction) QD_D3plot_info, METH_NOARGS, "Read a state variable."},
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
