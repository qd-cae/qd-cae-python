
#ifndef KEYFILE_PY
#define KEYFILE_PY

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration
class KeyFile;

/* QD_KeyFile OBJECT */
typedef struct {
    QD_FEMFile femfile; // Base
    /* Type-specific fields go here. */
    KeyFile* keyFile;
} QD_KeyFile;

static void
QD_KeyFile_dealloc(QD_KeyFile* self);

static int
QD_KeyFile_init(QD_KeyFile *self, PyObject *args, PyObject *kwds);

static PyObject *
QD_KeyFile_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

static PyMethodDef QD_KeyFile_methods[] = {
//  {"get_id", (PyCFunction) QD_Element_get_elementID, METH_NOARGS, "Get the element id."},
 {NULL}  /* Sentinel */
};

/* QD_KeyFile_Type TYPE */
static PyTypeObject QD_KeyFile_Type = {
  PyVarObject_HEAD_INIT(NULL, 0)
  "QD_KeyFile",             /* tp_name */
  sizeof(QD_KeyFile),           /* tp_basicsize */
  0,                         /* tp_itemsize */
  (destructor) QD_KeyFile_dealloc, /* tp_dealloc */
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
  Py_TPFLAGS_DEFAULT,        /* tp_flags */
  "QD_KeyFile",                 /* tp_doc */
  0,                         /* tp_traverse */
  0,                         /* tp_clear */
  0,                         /* tp_richcompare */
  0,                         /* tp_weaklistoffset */
  0,                         /* tp_iter */
  0,                         /* tp_iternext */
  QD_KeyFile_methods,        /* tp_methods */
  0,                         /* tp_members */
  0,                         /* tp_getset */
  &QD_FEMFile_Type,          /* tp_base */
  0,                         /* tp_dict */
  0,                         /* tp_descr_get */
  0,                         /* tp_descr_set */
  0,                         /* tp_dictoffset */
  (initproc) QD_KeyFile_init,   /* tp_init */
  0,                         /* tp_alloc */
  QD_KeyFile_new,             /* tp_new */
};


#ifdef __cplusplus
}
#endif

#endif
