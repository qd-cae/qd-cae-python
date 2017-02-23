
#ifndef PART_PY
#define PART_PY

// forward declarations
class Part;

// includes
#include <Python.h>

extern "C" {

  /* QD_Part OBJECT */
  typedef struct {
      PyObject_HEAD //;
      /* Type-specific fields go here. */
      Part* part;
      QD_FEMFile* femFile_py;
  } QD_Part;


  /* QD_Part DEALLOC */
  static void
  QD_Part_dealloc(QD_Part* self);

  /* QD_Part NEW */
  static PyObject *
  QD_Part_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

  /* QD_Part INIT */
  static int
  QD_Part_init(QD_Part *self, PyObject *args, PyObject *kwds);

  /* QD_Part FUNCTION get_id */
  static PyObject*
  QD_Part_get_id(QD_Part *self);

  /* QD_Part FUNCTION get_name */
  static PyObject*
  QD_Part_get_name(QD_Part *self);

  /* QD_Part FUNCTION get_nodes */
  static PyObject*
  QD_Part_get_nodes(QD_Part *self);

  /* QD_Part FUNCTION get_elements */
  static PyObject*
  QD_Part_get_elements(QD_Part *self, PyObject *args);

  /* QD_Part METHOD TABLE */
  static PyMethodDef QD_Part_methods[] = {
    {"get_id", (PyCFunction) QD_Part_get_id, METH_NOARGS, "Get the id of the part."},
    {"get_name", (PyCFunction) QD_Part_get_name, METH_NOARGS, "Get the name of the part."},
    {"get_nodes", (PyCFunction) QD_Part_get_nodes, METH_NOARGS, "Get the nodes of the part."},
    {"get_elements", (PyCFunction) QD_Part_get_elements, METH_VARARGS, "Get the elements of the part."},
    {NULL}  /* Sentinel */
  };


  /* QD_Part TYPE ... whatever */
  static PyTypeObject QD_Part_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "QD_Part",                 /* tp_name */
    sizeof(QD_Part),           /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)QD_Part_dealloc, /* tp_dealloc */
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
    "QD_Part",                    /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    QD_Part_methods,           /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc) QD_Part_init,   /* tp_init */
    0,                         /* tp_alloc */
    QD_Part_new,               /* tp_new */
  };

}

#endif
