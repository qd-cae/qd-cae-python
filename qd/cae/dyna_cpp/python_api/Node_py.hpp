
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
  QD_Node_richcompare(QD_Node *self, PyObject *other, int op);

  static PyObject *
  QD_Node_get_NodeID(QD_Node* self);

  const char* get_node_id_docs = "\
get_id()\n\
\n\
Get the id of the node.\n\
\n\
Returns\n\
-------\n\
id : int\n\
    id of the node\n\
\n\
Examples\n\
--------\n\
    >>> d3plot.get_nodeByID(1).get_id()\n\
    1\n\
";

  static PyObject *
  QD_Node_get_coords(QD_Node* self, PyObject *args, PyObject *kwds);

  const char* get_node_coords_docs = "\
get_coords(iTimestep=0)\n\
\n\
Get the geometric nodal coordinates at a timestep. One needs to load the displacements\n\
before getting the coordinates at a different timestep.\n\
\n\
Parameters\n\
----------\n\
iTimestep : int\n\
    timestep at which to take the coordinates\n\
\n\
Returns\n\
-------\n\
coords : np.ndarray\n\
    coordinate vector (x,z,y)\n\
\n\
Examples\n\
--------\n\
    >>> d3plot = D3plot(\"path/to/d3plot\")\n\
    >>> node.get_coords().shape\n\
    (3L,)\n\
    >>> # load disp\n\
    >>> d3plot.read_states(\"disp\")\n\
    >>> node.get_coords(iTimestep=10)\n\
";

  static PyObject *
  QD_Node_get_disp(QD_Node* self);

  const char* get_node_disp_docs = "\
get_disp()\n\
\n\
Get the time series of the displacement vector.\n\
\n\
Returns\n\
-------\n\
disp : np.ndarray\n\
    time series of displacements\n\
\n\
Examples\n\
--------\n\
    >>> node.get_disp().shape\n\
    (34L, 3L)\n\
";

  static PyObject *
  QD_Node_get_vel(QD_Node* self);

  const char* get_node_vel_docs = "\
get_vel()\n\
\n\
Get the time series of the velocity vector.\n\
\n\
Returns\n\
-------\n\
disp : np.ndarray\n\
    time series of displacements\n\
\n\
Examples\n\
--------\n\
    >>> node.get_disp().shape\n\
    (34L, 3L)\n\
";

  static PyObject *
  QD_Node_get_accel(QD_Node* self);

  const char* get_node_accel_docs = "\
get_accel()\n\
\n\
Get the time series of the acceleration vector.\n\
\n\
Returns\n\
-------\n\
disp : np.ndarray\n\
    time series of acceleration\n\
\n\
Examples\n\
--------\n\
    >>> node.get_accel().shape\n\
    (34L, 3L)\n\
";

  static PyObject *
  QD_Node_get_elements(QD_Node* self);

  const char* get_node_elements_docs = "\
get_elements()\n\
\n\
Get the elements of the node.\n\
\n\
Returns\n\
-------\n\
elements : list(Element)\n\
    elements of the node\n\
\n\
Examples\n\
--------\n\
    >>> len( node.get_elements() )\n\
    4\n\
";

  static PyMethodDef QD_Node_methods[] = {
    {"get_id", (PyCFunction) QD_Node_get_NodeID, METH_NOARGS, get_node_id_docs},
    {"get_coords", (PyCFunction) QD_Node_get_coords, METH_VARARGS | METH_KEYWORDS, get_node_coords_docs},
    {"get_disp", (PyCFunction) QD_Node_get_disp, METH_NOARGS, get_node_disp_docs},
    {"get_vel", (PyCFunction) QD_Node_get_vel, METH_NOARGS, get_node_vel_docs},
    {"get_accel", (PyCFunction) QD_Node_get_accel, METH_NOARGS, get_node_accel_docs},
    {"get_elements", (PyCFunction) QD_Node_get_elements, METH_NOARGS, get_node_elements_docs},
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
    "QD_Node",                 /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    (richcmpfunc)& QD_Node_richcompare, /* tp_richcompare */
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
