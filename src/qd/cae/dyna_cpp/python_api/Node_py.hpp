
#ifndef NODE_PY
#define NODE_PY

#ifdef __cplusplus
extern "C" {
#endif

  // Forward declaration
  class Node;
  class Element;
  class D3plot;

  /* QD_Node OBJECT */
  typedef struct {
      PyObject_HEAD //;
      /* Type-specific fields go here. */
      Node* node;
      QD_FEMFile* femFile_py;
  } QD_Node;

  static void
  QD_Node_dealloc(QD_Node* self);

  static int
  QD_Node_init(QD_Node *self, PyObject *args, PyObject *kwds);

  static PyObject *
  QD_Node_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

  static PyObject *
  QD_Node_get_NodeID(QD_Node* self);

  static PyObject *
  QD_Node_get_coords(QD_Node* self, PyObject *args, PyObject *kwds);

  static PyObject *
  QD_Node_get_disp(QD_Node* self);

  static PyObject *
  QD_Node_get_vel(QD_Node* self);

  static PyObject *
  QD_Node_get_accel(QD_Node* self);

  static PyObject *
  QD_Node_get_elements(QD_Node* self);

  static PyMethodDef QD_Node_methods[] = {
    {"get_id", (PyCFunction) QD_Node_get_NodeID, METH_NOARGS, "Get the node id."},
    {"get_coords", (PyCFunction) QD_Node_get_coords, METH_VARARGS, "Get the node coordinates."},
    {"get_disp", (PyCFunction) QD_Node_get_disp, METH_NOARGS, "Get the node displacement over time."},
    {"get_vel", (PyCFunction) QD_Node_get_vel, METH_NOARGS, "Get the node velocity over time."},
    {"get_accel", (PyCFunction) QD_Node_get_accel, METH_NOARGS, "Get the node acceleration over time."},
    {"get_elements", (PyCFunction) QD_Node_get_elements, METH_NOARGS, "Get the elements to which the node belongs."},
    {NULL}  /* Sentinel */
  };

  /* QD_Node_Type TYPE */
  static PyTypeObject QD_Node_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Node",             /* tp_name */
    sizeof(QD_Node),           /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor) QD_Node_dealloc, /* tp_dealloc */
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
    "Node",                 /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    QD_Node_methods,           /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc) QD_Node_init,   /* tp_init */
    0,                         /* tp_alloc */
    QD_Node_new,             /* tp_new */
  };


#ifdef __cplusplus
}
#endif

#endif
