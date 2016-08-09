
#ifndef NODE_PY
#define NODE_PY

#ifdef __cplusplus
extern "C" {
#endif

  // Forward declaration
  class Node;
  class Element;
  class D3plot;

  /* CD_Node OBJECT */
  typedef struct {
      PyObject_HEAD //;
      /* Type-specific fields go here. */
      Node* node;
      CD_D3plot* d3plot_py;
  } CD_Node;

  static void
  CD_Node_dealloc(CD_Node* self);

  static int
  CD_Node_init(CD_Node *self, PyObject *args, PyObject *kwds);

  static PyObject *
  CD_Node_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

  static PyObject *
  CD_Node_get_NodeID(CD_Node* self);

  static PyObject *
  CD_Node_get_coords(CD_Node* self, PyObject *args, PyObject *kwds);

  static PyObject *
  CD_Node_get_disp(CD_Node* self);

  static PyObject *
  CD_Node_get_vel(CD_Node* self);

  static PyObject *
  CD_Node_get_accel(CD_Node* self);

  static PyObject *
  CD_Node_get_elements(CD_Node* self);

  static PyMethodDef CD_Node_methods[] = {
    {"get_id", (PyCFunction) CD_Node_get_NodeID, METH_NOARGS, "Get the node id."},
    {"get_coords", (PyCFunction) CD_Node_get_coords, METH_VARARGS, "Get the node coordinates."},
    {"get_disp", (PyCFunction) CD_Node_get_disp, METH_NOARGS, "Get the node displacement over time."},
    {"get_vel", (PyCFunction) CD_Node_get_vel, METH_NOARGS, "Get the node velocity over time."},
    {"get_accel", (PyCFunction) CD_Node_get_accel, METH_NOARGS, "Get the node acceleration over time."},
    {"get_elements", (PyCFunction) CD_Node_get_elements, METH_NOARGS, "Get the elements to which the node belongs."},
    {NULL}  /* Sentinel */
  };

  /* CD_Node_Type TYPE */
  static PyTypeObject CD_Node_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Node",             /* tp_name */
    sizeof(CD_Node),           /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor) CD_Node_dealloc, /* tp_dealloc */
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
    CD_Node_methods,           /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc) CD_Node_init,   /* tp_init */
    0,                         /* tp_alloc */
    CD_Node_new,             /* tp_new */
  };


#ifdef __cplusplus
}
#endif

#endif
