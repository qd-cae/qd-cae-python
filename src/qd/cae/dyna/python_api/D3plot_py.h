
#ifndef D3PLOT_PY
#define D3PLOT_PY

#include <Python.h>
#include "../dyna/d3plot.h"

using namespace std;

extern "C" {

  /* CD_D3plot OBJECT */
  typedef struct {
      PyObject_HEAD //;
      /* Type-specific fields go here. */
      D3plot* d3plot;
  } CD_D3plot;


  /* CD_D3plot DEALLOC */
  static void
  CD_D3plot_dealloc(CD_D3plot* self);

  /* CD_D3plot NEW */
  static PyObject *
  CD_D3plot_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

  /* CD_D3plot INIT */
  static int
  CD_D3plot_init(CD_D3plot *self, PyObject *args, PyObject *kwds);

  /* CD_D3plot FUNCTION get_timesteps */
  static PyObject *
  CD_D3plot_get_timesteps(CD_D3plot* self);

  /* CD_D3plot FUNCTION get_filepath */
  static PyObject *
  CD_D3plot_get_filepath(CD_D3plot* self);

  /* CD_D3plot FUNCTION read_states */
  static PyObject *
  CD_D3plot_read_states(CD_D3plot* self, PyObject* args);

  /* CD_D3plot FUNCTION get_nodeByID */
  static PyObject *
  CD_D3plot_get_nodeByID(CD_D3plot* self, PyObject* args);

  /* CD_D3plot FUNCTION get_elementByID */
  static PyObject *
  CD_D3plot_get_elementByID(CD_D3plot* self, PyObject* args);

  /* CD_D3plot FUNCTION get_partByID */
  static PyObject *
  CD_D3plot_get_partByID(CD_D3plot* self, PyObject* args);
  
  /* CD_D3plot FUNCTION get_parts */
  static PyObject *
  CD_D3plot_get_parts(CD_D3plot* self, PyObject* args);

  /* CD_D3plot METHOD TABLE */
  static PyMethodDef CD_D3plot_methods[] = {
    {"get_timesteps", (PyCFunction) CD_D3plot_get_timesteps, METH_NOARGS, "Get the timesteps in the d3plot."},
    {"get_filepath", (PyCFunction) CD_D3plot_get_filepath, METH_NOARGS, "Get the filepath of the d3plot."},
    {"read_states", (PyCFunction) CD_D3plot_read_states, METH_VARARGS, "Read a state variable."},
    {"get_nodeByID", (PyCFunction) CD_D3plot_get_nodeByID, METH_VARARGS, "Get a node from it's id."},
    {"get_elementByID", (PyCFunction) CD_D3plot_get_elementByID, METH_VARARGS, "Get an element from it's id."},
    {"get_partByID", (PyCFunction) CD_D3plot_get_partByID, METH_VARARGS, "Get a part from it's id."},
    {"get_parts", (PyCFunction) CD_D3plot_get_parts, METH_NOARGS, "Get all the parts in a list."},
	{NULL}  /* Sentinel */
  };


  /* CD_D3plot TYPE ... whatever */
  static PyTypeObject CD_D3plot_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "D3plot",               /* tp_name */
    sizeof(CD_D3plot),         /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)CD_D3plot_dealloc, /* tp_dealloc */
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
        Py_TPFLAGS_BASETYPE,   /* tp_flags */
    "D3plot",               /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    CD_D3plot_methods,         /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc) CD_D3plot_init, /* tp_init */
    0,                         /* tp_alloc */
    CD_D3plot_new,             /* tp_new */
  };

}

#endif
