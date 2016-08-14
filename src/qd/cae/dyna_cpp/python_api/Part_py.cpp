

/* QD_Part DEALLOC */
static void
QD_Part_dealloc(QD_Part* self){
  Py_DECREF(self->d3plot_py);
}


/* QD_Part NEW */
static PyObject *
QD_Part_new(PyTypeObject *type, PyObject *args, PyObject *kwds){

  QD_Part* self;
  self = (QD_Part *)type->tp_alloc(type, 0);

  // Init vars if any ...
  if (self != NULL){
    self->part = NULL;
  }


  return (PyObject*) self;

}

/* QD_Part INIT */
static int
QD_Part_init(QD_Part *self, PyObject *args, PyObject *kwds){

  PyObject* d3plot_obj_py;
  int partID;
  static char *kwlist[] = {"d3plot","partID", NULL}; // TODO Deprecated!


  if (! PyArg_ParseTupleAndKeywords(args, kwds, "Oi", kwlist, &d3plot_obj_py, &partID)){
      return -1;
  }

  if (! PyObject_TypeCheck(d3plot_obj_py, &QD_D3plot_Type)) {
    PyErr_SetString(PyExc_TypeError, "arg #1 not a d3plot in part constructor");
    return -1;
  }
  QD_D3plot* d3plot_py = (QD_D3plot*) d3plot_obj_py;

  if(d3plot_py->d3plot == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to d3plot-object is NULL.");
    return -1;
  }

  self->d3plot_py = d3plot_py;
  Py_INCREF(self->d3plot_py);
  self->part = d3plot_py->d3plot->get_db_parts()->get_part_byID(partID);

  if(self->part == NULL){
    PyErr_SetString(PyExc_RuntimeError,string("Could not find any part with ID: "+to_string(partID)+".").c_str());
    return -1;
  }

  return 0;

}


static PyObject*
QD_Part_get_id(QD_Part *self){

  if(self->part == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to part is NULL.");
    return NULL;
  }

  return Py_BuildValue("i",self->part->get_partID());

}


/* QD_Part FUNCTION get_name */
static PyObject*
QD_Part_get_name(QD_Part *self){

  if(self->part == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to part is NULL.");
    return NULL;
  }

  string partName = self->part->get_name();

  return Py_BuildValue("s",partName.c_str());

}


/* QD_Part FUNCTION get_nodes */
static PyObject*
QD_Part_get_nodes(QD_Part *self){

  if(self->part == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to part is NULL.");
    return NULL;
  }

  set<Node*> nodes = self->part->get_nodes();

  int check=0;
  PyObject* node_list = PyList_New(nodes.size());

  unsigned int ii=0;
//   for(auto node : nodes){ // -std=c++11 rulez
  Node* node = NULL;
  for(set<Node*>::iterator it=nodes.begin(); it != nodes.end(); it++){
    node = *it;

    PyObject *argList2 = Py_BuildValue("Oi",self->d3plot_py ,node->get_nodeID());
    PyObject* ret = PyObject_CallObject((PyObject *) &QD_Node_Type, argList2);
    Py_DECREF(argList2);

    check += PyList_SetItem(node_list, ii, ret);

    ii++;
  }

  if(check != 0){
    PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of node instance list.");
    Py_DECREF(node_list);
    return NULL;
  }

  return node_list;

}


/* QD_Part FUNCTION get_elements */
static PyObject*
QD_Part_get_elements(QD_Part *self){

  if(self->part == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to part is NULL.");
    return NULL;
  }

  set<Element*> elements = self->part->get_elements();

  set<Element*>::iterator it;
  int check=0;
  PyObject* element_list = PyList_New(elements.size());

  unsigned int ii=0;
//   for(auto element : elements){ // -std=c++11 rulez
  Element* element = NULL;
  for(set<Element*>::iterator it=elements.begin(); it != elements.end(); it++){
    element = *it;

    PyObject* elementType_py;
    if(element->get_elementType() == 1){
      elementType_py = Py_BuildValue("s","beam");
    } else if(element->get_elementType() == 2) {
      elementType_py = Py_BuildValue("s","shell");
    } else if(element->get_elementType() == 3) {
      elementType_py = Py_BuildValue("s","solid");
    } else {
      PyErr_SetString(PyExc_SyntaxError, "Developer Error, unknown element-type.");
      return NULL;
    }

    PyObject *argList2 = Py_BuildValue("OOi",self->d3plot_py, elementType_py, element->get_elementID());
    PyObject* ret = PyObject_CallObject((PyObject *) &QD_Element_Type, argList2);
    Py_DECREF(argList2);
   Py_DECREF(elementType_py);

    check += PyList_SetItem(element_list, ii, ret);

    ii++;
  }

  if(check != 0){
    PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of element instance list.");
    Py_DECREF(element_list);
    return NULL;
  }

  return element_list;

}
