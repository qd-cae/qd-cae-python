

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

  const char* get_nodeByID_docs = "\
get_nodeByID(node_id)\n\
\n\
Parameters\n\
----------\n\
node_id : int\n\
    id of the node\n\
\n\
Returns\n\
-------\n\
node : Node\n\
    Node object\n\
\n\
Raises:\n\
-------\n\
ValueError\n\
    if `node_id` does not exist.\n\
\n\
Examples\n\
--------\n\
    >>> # get by single id\n\
    >>> node = femfile.get_nodeByID(1)\n\
    >>> # get a list of nodes at once\n\
    >>> list_of_nodes = femfile.get_nodeByID( [1,2,3] )\n\
";

  /* FUNCTION get_nodeByIndex */
  static PyObject *
  QD_FEMFile_get_nodeByIndex(QD_FEMFile* self, PyObject* args);

  const char* get_nodeByIndex_docs = "\
get_nodeByIndex(node_index)\n\
\n\
Parameters\n\
----------\n\
node_index : int\n\
    internal index of the node\n\
\n\
Returns\n\
-------\n\
node : Node\n\
    Node object\n\
\n\
Notes\n\
-----\n\
    The internal index starts at 0 and ends at\n\
    `femfile.get_nNodes()`.\n\
\n\
Raises:\n\
-------\n\
ValueError\n\
    if `node_index` larger `femfile.get_nNodes()`.\n\
\n\
Examples\n\
--------\n\
    >>> # single index\n\
    >>> node = femfile.get_nodeByIndex(1)\n\
    >>> # get a list of nodes at once\n\
    >>> list_of_nodes = femfile.get_nodeByIndex( [1,2,3] )\n\
";

  /* FUNCTION get_elementByID */
  static PyObject *
  QD_FEMFile_get_elementByID(QD_FEMFile* self, PyObject* args);

  const char* get_elementByID_docs = "\
get_elementByID(element_type, element_id)\n\
\n\
Parameters\n\
----------\n\
element_type : str\n\
    type of the element. Must be beam, shell or solid.\n\
element_id : int\n\
    id of the part in the file\n\
\n\
Returns\n\
-------\n\
part : Part\n\
    Part object\n\
\n\
Notes\n\
-----\n\
    Since ids in the dyna file are non unique for\n\
    different element types, one has to specify the\n\
    type too.\n\
\n\
Raises:\n\
-------\n\
ValueError\n\
    if `element_type` is invalid or `element_id` does not exist.\n\
\n\
Examples\n\
--------\n\
    >>> elem = femfile.get_elementByID(\"shell\",1)\n\
    >>> list_of_shells = femfile.get_elementByID(\"shell\", [1,2,3])\n\
    >>> # whoever had the great id of non unique ids ...\n\
    >>> femfile.get_elementByID(\"beam\", 1).get_type()\n\
    \"beam\"\n\
    >>> femfile.get_elementByID(\"solid\",1).get_type()\n\
    \"solid\"\n\
";

  /* FUNCTION get_partByID */
  static PyObject *
  QD_FEMFile_get_partByID(QD_FEMFile* self, PyObject* args);

  const char* get_partByID_docs = "get_partByID(part_id)\n\
\n\
Parameters\n\
----------\n\
part_id : int\n\
    id of the part in the file\n\
\n\
Returns\n\
-------\n\
part : Part\n\
    Part object\n\
\n\
Raises:\n\
-------\n\
ValueError\n\
    if `part_id` does not exist.\n\
\n\
Examples\n\
--------\n\
    >>> part = femfile.get_partByID(1)\n\
";

  /* FUNCTION get_parts */
  static PyObject *
  QD_FEMFile_get_parts(QD_FEMFile* self);

  const char* get_parts_docs = "\
get_parts()\n\
\n\
Returns\n\
-------\n\
parts : list(Part)\n\
    list of all parts in the file\n\
\n\
Examples\n\
--------\n\
    >>> list_of_all_parts = femfile.get_parts()\n\
";

  /* FUNCTION get_mesh */
  static PyObject *
  QD_FEMFile_get_mesh(QD_FEMFile* self, PyObject* args);

  const char* get_mesh_docs = "get_mesh()\n\
\n\
unfinished.\n\
";

  /* FUNCTON get_nNodes */
  static PyObject *
  QD_FEMFile_get_nNodes(QD_FEMFile* self);

  const char* get_nNodes_docs = "\
get_nNodes()\n\
\n\
Returns\n\
-------\n\
nNodes : int\n\
    number of nodes in the file\n\
\n\
Examples\n\
--------\n\
    >>> femfile.get_nNodes()\n\
    43145\n\
";

  /* FUNCTON get_nodes */
  static PyObject *
  QD_FEMFile_get_nodes(QD_FEMFile* self);

  const char* get_nodes_docs = "\
get_nodes()\n\
\n\
Returns\n\
-------\n\
nodes : list(Node)\n\
    list of node objects\n\
\n\
Examples\n\
--------\n\
    >>> list_of_nodes = femfile.get_nodes()\n\
";

  /* FUNCTON get_nElements */
  static PyObject *
  QD_FEMFile_get_nElements(QD_FEMFile* self, PyObject* args);

  const char* get_nElements_docs = "get_nElements(element_type=None)\n\
\n\
Parameters\n\
----------\n\
element_type : str\n\
    Optional element type filter. May be beam, shell or solid.\n\
\n\
Returns\n\
-------\n\
nElements : int\n\
    number of elements\n\
\n\
Examples\n\
--------\n\
    >>> femfile.get_nElements()\n\
    43156\n\
";

  /* FUNCTON get_nElements */
  static PyObject *
  QD_FEMFile_get_elements(QD_FEMFile* self, PyObject* args);

  const char* get_elements_docs = "\
get_elements(element_type=None)\n\
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
Raises:\n\
-------\n\
ValueError\n\
    if invalid `element_type` filter.\n\
\n\
Notes\n\
-----\n\
Get the elements of the femfile. One may use a filter by type.\n\
\n\
Examples\n\
--------\n\
    >>> all_elements = femfile.get_elements()\n\
    >>> shell_elements = femfile.get_elements('shell')\n\
";

  /* FUNCTION get_filepath */
  static PyObject *
  QD_FEMFile_get_filepath(QD_FEMFile* self);

  const char* get_filepath_docs = "\
get_filepath()\n\
\n\
Returns\n\
-------\n\
filepath : str\n\
    Filepath of the femfile.\n\
\n\
Examples\n\
--------\n\
    >>> femfile.get_filepath()\n\
    \"path/to/femfile\"\n\
";

  static PyMethodDef QD_FEMFile_methods[] = {
   {"get_filepath", (PyCFunction) QD_FEMFile_get_filepath, METH_NOARGS, get_filepath_docs},
   {"get_nodeByID", (PyCFunction) QD_FEMFile_get_nodeByID, METH_VARARGS, get_nodeByID_docs},
   {"get_nodeByIndex", (PyCFunction) QD_FEMFile_get_nodeByIndex, METH_VARARGS, get_nodeByIndex_docs},
   {"get_elementByID", (PyCFunction) QD_FEMFile_get_elementByID, METH_VARARGS, get_elementByID_docs},
   {"get_partByID", (PyCFunction) QD_FEMFile_get_partByID, METH_VARARGS, get_partByID_docs},
   {"get_parts", (PyCFunction) QD_FEMFile_get_parts, METH_NOARGS, get_parts_docs},
   {"get_nodes", (PyCFunction) QD_FEMFile_get_nodes, METH_NOARGS, get_nodes_docs},
   //{"get_elements", (PyCFunction) QD_FEMFile_get_elements, METH_VARARGS, "Get all the elements of the femfile in a list."},
   {"get_elements", (PyCFunction) QD_FEMFile_get_elements, METH_VARARGS, get_elements_docs},
   {"get_mesh", (PyCFunction) QD_FEMFile_get_mesh, METH_VARARGS, get_mesh_docs},
   {"get_nNodes", (PyCFunction) QD_FEMFile_get_nNodes, METH_NOARGS, get_nNodes_docs},
   {"get_nElements", (PyCFunction) QD_FEMFile_get_nElements, METH_VARARGS, get_nElements_docs},
   {NULL}  /* Sentinel */
  };

  const char* femfile_docs = "\
This class is a super class of the D3plot and the KeyFile. It is meant \
for handling the mesh data.\n\
\n\
>>> from qd.cae.dyna import D3plot, FEMFile, KeyFile\n\
>>> issubclass(D3plot, FEMFile)\n\
True\n\
>>> issubclass(KeyFile, FEMFile)\n\
True\n\
";

  /* TYPE */
  static PyTypeObject QD_FEMFile_Type = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "FEMFile",                 /* tp_name */
    sizeof(QD_FEMFile),        /* tp_basicsize */
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
    femfile_docs,                 /* tp_doc */
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
