
#ifndef ELEMENT_PY
#define ELEMENT_PY

#ifdef __cplusplus
extern "C" {
#endif

  // Forward declaration
  class Node;
  class D3plot;
  class Element;

  /* QD_Element OBJECT */
  typedef struct {
      PyObject_HEAD //;
      /* Type-specific fields go here. */
      Element* element;
      QD_FEMFile* femFile_py;
  } QD_Element;

  static void
  QD_Element_dealloc(QD_Element* self);

  static int
  QD_Element_init(QD_Element *self, PyObject *args, PyObject *kwds);

  static PyObject *
  QD_Element_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

  static PyObject *
  QD_Element_richcompare(QD_Element *self, PyObject *other, int op);

  static PyObject *
  QD_Element_get_elementID(QD_Element* self);

  const char* get_elementID_docs = "\
get_id()\n\
\n\
Get the id of the element.\n\
\n\
Returns\n\
-------\n\
id : int\n\
    id of the element\n\
\n\
Examples\n\
--------\n\
    >>> d3plot.get_elementByID(\"shell\",1).get_id()\n\
    1\n\
";

  static PyObject *
  QD_Element_get_plastic_strain(QD_Element* self);

  const char* get_plastic_strain_docs = "\
get_plastic_strain()\n\
\n\
Get the plastic strain of the element, if it was read with `read_states`.\n\
\n\
Returns\n\
-------\n\
plastic_strain : np.ndarray\n\
    time series of plastic strain\n\
\n\
Examples\n\
--------\n\
    >>> element.get_plastic_strain().shape\n\
    (34L,)\n\
";

  static PyObject *
  QD_Element_get_energy(QD_Element* self);

  const char* get_energy_docs = "\
get_energy()\n\
\n\
Get the energy of the element, if it was read with `read_states`.\n\
\n\
Returns\n\
-------\n\
energy : np.ndarray\n\
    time series of element energy\n\
\n\
Examples\n\
--------\n\
    >>> element.get_energy().shape\n\
    (34L,)\n\
";

  static PyObject *
  QD_Element_get_strain(QD_Element* self);

  const char* get_strain_docs = "\
get_strain()\n\
\n\
Get the strain tensor of the element, if it was read with `read_states`.\n\
The strain vector contains the matrix components: [e_xx, e_yy, e_zz, e_xy, e_yz, e_xz]\n\
\n\
Returns\n\
-------\n\
strain : np.ndarray\n\
    time series of the strain tensor data\n\
\n\
Examples\n\
--------\n\
    >>> element.get_strain().shape\n\
    (34L, 6L)\n\
";

  static PyObject *
  QD_Element_get_stress(QD_Element* self);

  const char* get_stress_docs = "\
get_strain()\n\
\n\
Get the stress tensor of the element, if it was read with `read_states`.\n\
The stress vector contains the matrix components: [s_xx, s_yy, s_zz, s_xy, s_yz, s_xz]\n\
\n\
Returns\n\
-------\n\
stress : np.ndarray\n\
    time series of the stress tensor data\n\
\n\
Examples\n\
--------\n\
    >>> element.get_stress().shape\n\
    (34L, 6L)\n\
";

  static PyObject *
  QD_Element_get_stress_mises(QD_Element* self);

  const char* get_stress_mises_docs = "\
get_stress_mises()\n\
\n\
Get the mises stress of the element, if it was read with `read_states`.\n\
\n\
Returns\n\
-------\n\
stress : np.ndarray\n\
    time series of the mises stress\n\
\n\
Examples\n\
--------\n\
    >>> element.get_stress_mises().shape\n\
    (34L,)\n\
";

  static PyObject *
  QD_Element_get_nodes(QD_Element* self);

  const char* get_element_nodes_docs = "\
get_nodes()\n\
\n\
Get the nodes of the element.\n\
\n\
Returns\n\
-------\n\
nodes : list(Node)\n\
    nodes of the element\n\
\n\
Examples\n\
--------\n\
    >>> elem_nodes = element.get_nodes()\n\
";

  static PyObject *
  QD_Element_get_coords(QD_Element* self, PyObject *args, PyObject *kwds);

  const char* get_element_coords_docs = "\
get_coords(iTimestep=0)\n\
\n\
Get the elements coordinates (mean of nodes).\n\
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
    >>> element.get_coords().shape\n\
    (3L,)\n\
    >>> some_coords = element.get_coords(iTimestep=10) # disps must be loaded\n\
";

  static PyObject *
  QD_Element_get_history(QD_Element* self);

  const char* get_history_docs = "\
get_history()\n\
\n\
Get the loaded history variables of the element.\n\
\n\
Returns\n\
-------\n\
history_vars : np.ndarray\n\
    time series of the loaded history variables\n\
\n\
Examples\n\
--------\n\
    >>> d3plot = D3plot(\"path/to/d3plot\",read_states=\"history 1 shell max\")\n\
    >>> d3plot.get_elementByID(\"shell\",1).get_history().shape\n\
    (34L, 1L)\n\
\n\
Notes\n\
-----\n\
    The history variable column index corresponds to the order in which\n\
    the variables were loaded\n\
";

  static PyObject *
  QD_Element_get_estimated_size(QD_Element* self);

  const char* get_estimated_size_docs = "\
get_estimated_size()\n\
\n\
Get the average element edge size of the element.\n\
\n\
Returns\n\
-------\n\
size : float\n\
    average element edge size\n\
\n\
Examples\n\
--------\n\
    >>> element.get_estimated_size()\n\
    2.542\n\
";

  static PyObject *
  QD_Element_get_type(QD_Element* self);

  const char* get_type_docs = "\
get_type()\n\
\n\
Get the type of the element.\n\
\n\
Returns\n\
-------\n\
element_type : str\n\
    beam, shell or solid\n\
\n\
Examples\n\
--------\n\
    >>> d3plot.get_elementByID(\"beam\",1).get_type()\n\
    'beam'\n\
    >>> d3plot.get_elementByID(\"shell\",1).get_type()\n\
    'shell'\n\
    >>> d3plot.get_elementByID(\"solid\",1).get_type()\n\
    'solid'\n\
";

  static PyObject *
  QD_Element_get_is_rigid(QD_Element* self);

  const char* get_is_rigid_docs = "\
is_rigid()\n\
\n\
Get the status, whether the element is a rigid (flag for shells only).\n\
Rigid shells have no state data.\n\
\n\
Returns\n\
-------\n\
is_rigid : bool\n\
    rigid status of the element\n\
\n\
Examples\n\
--------\n\
    >>> d3plot = D3plot(\"path/to/d3plot\", read_states=\"stress_mises max\")\n\
    >>> elem1 = d3plot.get_elementByID(\"shell\", 451)\n\
    >>> elem1.is_rigid()\n\
    False\n\
    >>> elem1.get_stress_mises().shape\n\
    (34L,)\n\
    >>> elem2 = d3plot.get_elementByID(\"shell\", 9654)\n\
    >>> elem2.is_rigid()\n\
    True\n\
    >>> elem2.get_stress_mises().shape\n\
    (0L,)\n\
";

  static PyMethodDef QD_Element_methods[] = {
    {"get_id", (PyCFunction) QD_Element_get_elementID, METH_NOARGS, get_elementID_docs},
    {"get_plastic_strain", (PyCFunction) QD_Element_get_plastic_strain, METH_NOARGS, get_plastic_strain_docs},
    {"get_energy", (PyCFunction) QD_Element_get_energy, METH_NOARGS, get_energy_docs},
    {"get_strain", (PyCFunction) QD_Element_get_strain, METH_NOARGS, get_strain_docs},
    {"get_stress", (PyCFunction) QD_Element_get_stress, METH_NOARGS, get_stress_docs},
    {"get_stress_mises", (PyCFunction) QD_Element_get_stress_mises, METH_NOARGS, get_stress_mises_docs},
    {"get_nodes", (PyCFunction) QD_Element_get_nodes, METH_NOARGS, get_element_nodes_docs},
    {"get_coords", (PyCFunction) QD_Element_get_coords, METH_VARARGS, get_element_coords_docs},
    {"get_history", (PyCFunction) QD_Element_get_history, METH_NOARGS, get_history_docs},
    {"get_estimated_size", (PyCFunction) QD_Element_get_estimated_size, METH_NOARGS, get_estimated_size_docs},
    {"get_type", (PyCFunction) QD_Element_get_type, METH_NOARGS, get_type_docs},
    {"is_rigid", (PyCFunction) QD_Element_get_is_rigid, METH_NOARGS, get_is_rigid_docs},
    {NULL}  /* Sentinel */
  };

  const char* qd_element_class_docs = "\
\n\
Container for handling element data.\n\
\n\
Parameters\n\
----------\n\
femfile : FEMFile\n\
    femfile from which to take the Element\n\
element_type : str\n\
    beam, shell or solid\n\
id : int\n\
    id of the element in the file\n\
\n\
Examples\n\
--------\n\
    It is recommended to get elements by\n\
\n\
    >>> femfile = D3plot(\"path/to/d3plot\")\n\
    >>> element_list = femfile.get_elements()\n\
    >>> shells = femfile.get_elements(\"shell\")\n\
    >>> id = 1\n\
    >>> element = femfile.get_elementByID(\"solid\", id)\n\
";

  /* QD_Element_Type TYPE */
  static PyTypeObject QD_Element_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Element",             /* tp_name */
    sizeof(QD_Element),           /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor) QD_Element_dealloc, /* tp_dealloc */
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
    Py_TPFLAGS_BASETYPE,       /* tp_flags */
    qd_element_class_docs,                 /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    (richcmpfunc)& QD_Element_richcompare, /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    QD_Element_methods,           /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc) QD_Element_init,   /* tp_init */
    0,                         /* tp_alloc */
    QD_Element_new,             /* tp_new */
  };


#ifdef __cplusplus
}
#endif

#endif
