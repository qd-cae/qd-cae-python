
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

  static PyObject *
  QD_Element_get_plastic_strain(QD_Element* self);

  static PyObject *
  QD_Element_get_energy(QD_Element* self);

  static PyObject *
  QD_Element_get_strain(QD_Element* self);

  static PyObject *
  QD_Element_get_stress(QD_Element* self);

  static PyObject *
  QD_Element_get_nodes(QD_Element* self);

  static PyObject *
  QD_Element_get_coords(QD_Element* self, PyObject *args, PyObject *kwds);

  static PyObject *
  QD_Element_get_history(QD_Element* self);

  static PyObject *
  QD_Element_get_estimated_size(QD_Element* self);

  static PyObject *
  QD_Element_get_type(QD_Element* self);
  
  static PyObject *
  QD_Element_get_is_rigid(QD_Element* self);

  static PyMethodDef QD_Element_methods[] = {
    {"get_id", (PyCFunction) QD_Element_get_elementID, METH_NOARGS, "Get the element id."},
    {"get_plastic_strain", (PyCFunction) QD_Element_get_plastic_strain, METH_NOARGS, "Get the plastic strain time series of the element."},
    {"get_energy", (PyCFunction) QD_Element_get_energy, METH_NOARGS, "Get the energy time series of the element."},
    {"get_strain", (PyCFunction) QD_Element_get_strain, METH_NOARGS, "Get the strain time series of the element."},
    {"get_stress", (PyCFunction) QD_Element_get_stress, METH_NOARGS, "Get the stress time series of the element."},
    {"get_nodes", (PyCFunction) QD_Element_get_nodes, METH_NOARGS, "Get the nodes of the element."},
    {"get_coords", (PyCFunction) QD_Element_get_coords, METH_VARARGS, "Get the coords of the element at a given timestep."},
    {"get_history", (PyCFunction) QD_Element_get_history, METH_NOARGS, "Get the history vars of the element."},
    {"get_estimated_size", (PyCFunction) QD_Element_get_estimated_size, METH_NOARGS, "Get the rough element edge size."},
    {"get_type", (PyCFunction) QD_Element_get_type, METH_NOARGS, "Get the element type."},
    {"get_is_rigid", (PyCFunction) QD_Element_get_is_rigid, METH_NOARGS, "Get the info if the element is rigid."},
    {NULL}  /* Sentinel */
  };

  /* QD_Element_Type TYPE */
  static PyTypeObject QD_Element_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "QD_Element",             /* tp_name */
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
    "QD_Element",                 /* tp_doc */
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
