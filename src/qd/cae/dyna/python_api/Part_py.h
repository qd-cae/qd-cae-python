
#ifndef PART_PY
#define PART_PY

#include <Python.h>
#include "../db/Part.h"

using namespace std;

extern "C" {

  /* CD_Part OBJECT */
  typedef struct {
      PyObject_HEAD //;
      /* Type-specific fields go here. */
      Part* part;
      CD_D3plot* d3plot_py;
  } CD_Part;


  /* CD_Part DEALLOC */
  static void
  CD_Part_dealloc(CD_Part* self);

  /* CD_Part NEW */
  static PyObject *
  CD_Part_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

  /* CD_Part INIT */
  static int
  CD_Part_init(CD_Part *self, PyObject *args, PyObject *kwds);

  /* CD_Part FUNCTION get_id */
  static PyObject*
  CD_Part_get_id(CD_Part *self);

  /* CD_Part FUNCTION get_name */
  static PyObject*
  CD_Part_get_name(CD_Part *self);

  /* CD_Part FUNCTION get_nodes */
  static PyObject*
  CD_Part_get_nodes(CD_Part *self);

  /* CD_Part FUNCTION get_elements */
  static PyObject*
  CD_Part_get_elements(CD_Part *self);

  /* CD_Part METHOD TABLE */
  static PyMethodDef CD_Part_methods[] = {
    {"get_id", (PyCFunction) CD_Part_get_id, METH_NOARGS, "Get the id of the part."},
    {"get_name", (PyCFunction) CD_Part_get_name, METH_NOARGS, "Get the name of the part."},
    {"get_nodes", (PyCFunction) CD_Part_get_nodes, METH_NOARGS, "Get the nodes of the part."},
    {"get_elements", (PyCFunction) CD_Part_get_elements, METH_NOARGS, "Get the elements of the part."},
    {NULL}  /* Sentinel */
  };


  /* CD_Part TYPE ... whatever */
  static PyTypeObject CD_Part_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Part",                 /* tp_name */
    sizeof(CD_Part),           /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)CD_Part_dealloc, /* tp_dealloc */
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
    "Part",                    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    CD_Part_methods,           /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc) CD_Part_init,   /* tp_init */
    0,                         /* tp_alloc */
    CD_Part_new,               /* tp_new */
  };

}

#endif
