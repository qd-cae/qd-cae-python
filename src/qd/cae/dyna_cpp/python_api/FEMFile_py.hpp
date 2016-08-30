

#ifndef FEMFILE_PY
#define FEMFILE_PY

#ifdef __cplusplus
extern "C" {
#endif

  // Forward declaration
  class FEMFile;

  /* OBJECT */
  typedef struct {
      PyObject_HEAD //;
      /* Type-specific fields go here. */
      FEMFile* instance;
  } QD_FEMFile;

  static void
  QD_FEMFile_dealloc(QD_FEMFile* self);

  static int
  QD_FEMFile_init(QD_FEMFile *self, PyObject *args, PyObject *kwds);

  static PyObject *
  QD_FEMFile_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

  /* FUNCTION get_nodeByID */
  static PyObject *
  QD_FEMFile_get_nodeByID(QD_FEMFile* self, PyObject* args);

  /* FUNCTION get_elementByID */
  static PyObject *
  QD_FEMFile_get_elementByID(QD_FEMFile* self, PyObject* args);

  /* FUNCTION get_partByID */
  static PyObject *
  QD_FEMFile_get_partByID(QD_FEMFile* self, PyObject* args);

  /* FUNCTION get_parts */
  static PyObject *
  QD_FEMFile_get_parts(QD_FEMFile* self, PyObject* args);

  /* FUNCTION get_filepath */
  static PyObject *
  QD_FEMFile_get_filepath(QD_FEMFile* self);


  static PyMethodDef QD_FEMFile_methods[] = {
   {"get_filepath", (PyCFunction) QD_FEMFile_get_filepath, METH_NOARGS, "Get the filepath."},
   {"get_nodeByID", (PyCFunction) QD_FEMFile_get_nodeByID, METH_VARARGS, "Get a node from it's id."},
   {"get_elementByID", (PyCFunction) QD_FEMFile_get_elementByID, METH_VARARGS, "Get an element from it's id."},
   {"get_partByID", (PyCFunction) QD_FEMFile_get_partByID, METH_VARARGS, "Get a part from it's id."},
   {"get_parts", (PyCFunction) QD_FEMFile_get_parts, METH_NOARGS, "Get all the parts in a list."},
   {NULL}  /* Sentinel */
  };

  /* TYPE */
  static PyTypeObject QD_FEMFile_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "QD_FEMFile",             /* tp_name */
    sizeof(QD_FEMFile),           /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor) QD_FEMFile_dealloc, /* tp_dealloc */
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
    "QD_FEMFile",                 /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    QD_FEMFile_methods,           /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc) QD_FEMFile_init,   /* tp_init */
    0,                         /* tp_alloc */
    QD_FEMFile_new,             /* tp_new */
  };


#ifdef __cplusplus
}
#endif

#endif
