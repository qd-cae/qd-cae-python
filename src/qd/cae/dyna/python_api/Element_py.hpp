
#ifndef ELEMENT_PY
#define ELEMENT_PY

#ifdef __cplusplus
extern "C" {
#endif

  // Forward declaration
  class Node;
  class D3plot;
  class Element;

  /* CD_Element OBJECT */
  typedef struct {
      PyObject_HEAD //;
      /* Type-specific fields go here. */
      Element* element;
      CD_D3plot* d3plot_py;
  } CD_Element;

  static void
  CD_Element_dealloc(CD_Element* self);

  static int
  CD_Element_init(CD_Element *self, PyObject *args, PyObject *kwds);

  static PyObject *
  CD_Element_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

  static PyObject *
  CD_Element_get_elementID(CD_Element* self);

  static PyObject *
  CD_Element_get_plastic_strain(CD_Element* self);

  static PyObject *
  CD_Element_get_energy(CD_Element* self);

  static PyObject *
  CD_Element_get_strain(CD_Element* self);

  static PyObject *
  CD_Element_get_stress(CD_Element* self);

  static PyObject *
  CD_Element_get_nodes(CD_Element* self);
  
  static PyObject *
  CD_Element_get_coords(CD_Element* self, PyObject *args, PyObject *kwds);
  
  static PyObject *
  CD_Element_get_history(CD_Element* self);
  
  static PyObject *
  CD_Element_get_estimated_size(CD_Element* self);
  
  static PyObject *
  CD_Element_get_type(CD_Element* self);

  static PyMethodDef CD_Element_methods[] = {
    {"get_id", (PyCFunction) CD_Element_get_elementID, METH_NOARGS, "Get the element id."},
    {"get_plastic_strain", (PyCFunction) CD_Element_get_plastic_strain, METH_NOARGS, "Get the plastic strain time series of the element."},
    {"get_energy", (PyCFunction) CD_Element_get_energy, METH_NOARGS, "Get the energy time series of the element."},
    {"get_strain", (PyCFunction) CD_Element_get_strain, METH_NOARGS, "Get the strain time series of the element."},
    {"get_stress", (PyCFunction) CD_Element_get_stress, METH_NOARGS, "Get the stress time series of the element."},
    {"get_nodes", (PyCFunction) CD_Element_get_nodes, METH_NOARGS, "Get the nodes of the element."},
    {"get_coords", (PyCFunction) CD_Element_get_coords, METH_VARARGS, "Get the coords of the element at a given timestep."},
	 {"get_history", (PyCFunction) CD_Element_get_history, METH_NOARGS, "Get the history vars of the element."},
    {"get_estimated_size", (PyCFunction) CD_Element_get_estimated_size, METH_NOARGS, "Get the estimated size of the element."},
    {"get_type", (PyCFunction) CD_Element_get_type, METH_NOARGS, "Get the element type."},
   {NULL}  /* Sentinel */
  };

  /* CD_Element_Type TYPE */
  static PyTypeObject CD_Element_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "Element",             /* tp_name */
    sizeof(CD_Element),           /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor) CD_Element_dealloc, /* tp_dealloc */
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
    "Element",                 /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    CD_Element_methods,           /* tp_methods */
    0,                         /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc) CD_Element_init,   /* tp_init */
    0,                         /* tp_alloc */
    CD_Element_new,             /* tp_new */
  };


#ifdef __cplusplus
}
#endif

#endif
