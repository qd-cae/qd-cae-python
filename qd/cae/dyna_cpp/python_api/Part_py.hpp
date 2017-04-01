
#ifndef PART_PY
#define PART_PY

// forward declarations
class Part;

// includes
#include <Python.h>

extern "C" {

  /* OBJECT */
  typedef struct {
      PyObject_HEAD //;
      /* Type-specific fields go here. */
      Part* part;
      QD_FEMFile* femFile_py;
  } QD_Part;


  /* DEALLOC */
  static void
  QD_Part_dealloc(QD_Part* self);

  /* NEW */
  static PyObject *
  QD_Part_new(PyTypeObject *type, PyObject *args, PyObject *kwds);

  /* INIT */
  static int
  QD_Part_init(QD_Part *self, PyObject *args, PyObject *kwds);

  /* FUNCTION get_id */
  static PyObject * 
  QD_Part_richcompare(QD_Part *self, PyObject *other, int op);

  /* FUNCTION get_id */
  static PyObject*
  QD_Part_get_id(QD_Part *self);

  const char* get_part_id_docs = "\
get_id()\n\
\n\
Get the id of the part.\n\
\n\
Returns\n\
-------\n\
id : int\n\
    id of the part\n\
\n\
Examples\n\
--------\n\
    >>> d3plot = D3plot(\"path/to/d3plot\")\n\
    >>> part = d3plot.get_partByID(1)\n\
    >>> part.get_id()\n\
    1\n\
";

  /* FUNCTION get_name */
  static PyObject*
  QD_Part_get_name(QD_Part *self);

  const char* get_part_name_docs = "\
get_name()\n\
\n\
Get the name of the part. It's the same name as in the input deck.\n\
\n\
Returns\n\
-------\n\
name : str\n\
    name of the part\n\
\n\
Examples\n\
--------\n\
    >>> d3plot = D3plot(\"path/to/d3plot\")\n\
    >>> part = d3plot.get_partByID(1)\n\
    >>> part.get_name()\n\
    'PLATE_C'\n\
";

  /* FUNCTION get_nodes */
  static PyObject*
  QD_Part_get_nodes(QD_Part *self);

  const char* get_part_nodes_docs = "\
get_nodes()\n\
\n\
Get the nodes of the part. Note that a node may belong to two parts,\n\
since only the elements are uniquely assignable.\n\
\n\
Returns\n\
-------\n\
nodes : list(Node)\n\
    nodes belonging to the elements of the part\n\
\n\
Examples\n\
--------\n\
    >>> d3plot = D3plot(\"path/to/d3plot\")\n\
    >>> part = d3plot.get_partByID(1)\n\
    >>> len( part.get_nodes() )\n\
    52341\n\
";

  /* FUNCTION get_elements */
  static PyObject*
  QD_Part_get_elements(QD_Part *self, PyObject *args);

  const char* get_part_elements_docs = "\
get_elements(element_type=None)\n\
\n\
Get the elements of the part.\n\
\n\
Parameters\n\
----------\n\
element_type : str\n\
    Optional element type filter. May be beam, shell or solid.\n\
\n\
Returns\n\
-------\n\
elements : list(Element)\n\
    list of Elements\n\
\n\
Examples\n\
--------\n\
    >>> d3plot = D3plot(\"path/to/d3plot\")\n\
    >>> part = d3plot.get_partByID(1)\n\
    >>> len( part.get_elements() )\n\
    49123\n\
    >>> len( part.get_elements(\"shell\") )\n\
    45123\n\
";

  /* METHOD TABLE */
  static PyMethodDef QD_Part_methods[] = {
    {"get_id", (PyCFunction) QD_Part_get_id, METH_NOARGS, get_part_id_docs},
    {"get_name", (PyCFunction) QD_Part_get_name, METH_NOARGS, get_part_name_docs},
    {"get_nodes", (PyCFunction) QD_Part_get_nodes, METH_NOARGS, get_part_nodes_docs},
    {"get_elements", (PyCFunction) QD_Part_get_elements, METH_VARARGS, get_part_elements_docs},
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
    (richcmpfunc)& QD_Part_richcompare, /* tp_richcompare */
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
