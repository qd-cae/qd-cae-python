

/* QD_Element DEALLOC */
static void
QD_Element_dealloc(QD_Element* self)
{

  Py_DECREF(self->femFile_py);

}

/* QD_Element NEW */
static PyObject *
QD_Element_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

  QD_Element* self;
  self = (QD_Element *)type->tp_alloc(type, 0);

  // Init vars if any ...
  if (self != NULL){
    self->element = NULL;
  }


  return (PyObject*) self;

}


/* QD_Element INIT */
static int
QD_Element_init(QD_Element *self, PyObject *args, PyObject *kwds)
{

  PyObject* femfile_obj_py;
  char* elementType_c;
  int elementID;
  static char *kwlist[] = {"femfile","elementType","elementID", NULL}; // TODO Deprecated!

  if (! PyArg_ParseTupleAndKeywords(args, kwds, "Osi", kwlist, &femfile_obj_py, &elementType_c, &elementID)){
     return -1;
  }

  if (! PyObject_TypeCheck(femfile_obj_py, &QD_FEMFile_Type)) {
    PyErr_SetString(PyExc_TypeError, "arg #1 not a D3plot or KeyFile in element constructor");
    return -1;
  }

  int elementType;
  string elementType_s(elementType_c);
  if( elementType_s.find("beam") !=  string::npos){
    elementType = BEAM;
  } else if( elementType_s.find("shell") !=  string::npos){
    elementType = SHELL;
  } else if( elementType_s.find("solid") !=  string::npos){
    elementType = SOLID;
  } else {
    PyErr_SetString(PyExc_SyntaxError, "Unknown element-type. Try: beam, shell, solid.");
    return -1;
  }

  QD_FEMFile* femFile_py = (QD_FEMFile*) femfile_obj_py;

  if(femFile_py->instance == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to C++ File-Object is NULL.");
    return -1;
  }

  self->femFile_py = femFile_py;
  Py_INCREF(self->femFile_py);
  self->element = femFile_py->instance->get_db_elements()->get_elementByID(elementType,elementID);

  if(self->element == NULL){
    PyErr_SetString(PyExc_RuntimeError,string("Could not find any element with ID: "+to_string(elementID)+".").c_str());
    return -1;
  }

  return 0;
}


/* QD_Element FUNCTION get_id */
static PyObject *
QD_Element_get_elementID(QD_Element* self){

  if(self->element == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
    return NULL;
  }

  int elementID = self->element->get_elementID();

  return Py_BuildValue("i",elementID);
}


/* QD_Element FUNCTION get_plastic_strain */
static PyObject *
QD_Element_get_plastic_strain(QD_Element* self){

  if(self->element == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
    return NULL;
  }

  return (PyObject*) vector_to_nparray(self->element->get_plastic_strain());

}


/* QD_Element FUNCTION get_energy */
static PyObject *
QD_Element_get_energy(QD_Element* self){

  if(self->element == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
    return NULL;
  }

  return (PyObject*) vector_to_nparray(self->element->get_energy());

}


/* QD_Element FUNCTION get_strain */
static PyObject *
QD_Element_get_strain(QD_Element* self){

  if(self->element == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
    return NULL;
  }

  return (PyObject*) vector_to_nparray(self->element->get_strain());

}


/* QD_Element FUNCTION get_stress */
static PyObject *
QD_Element_get_stress(QD_Element* self){

  if(self->element == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
    return NULL;
  }

  return (PyObject*) vector_to_nparray(self->element->get_stress());

}

/* QD_Element FUNCTION get_nodes */
static PyObject *
QD_Element_get_nodes(QD_Element* self){

  if(self->element == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
    return NULL;
  }

  set<Node*> nodes = self->element->get_nodes();

  set<Node*>::iterator it;
  int check=0;
  PyObject* node_list = PyList_New(nodes.size());

  unsigned int ii=0;
  Node* node = NULL;
  for(set<Node*>::iterator it=nodes.begin(); it != nodes.end(); ++it){
    node = *it;

    PyObject *argList2 = Py_BuildValue("Oi",self->femFile_py ,node->get_nodeID());
    PyObject* ret = PyObject_CallObject((PyObject *) &QD_Node_Type, argList2);
    Py_DECREF(argList2);

    check += PyList_SetItem(node_list, ii, ret);

    ++ii;
  }

  if(check != 0){
    Py_DECREF(node_list);
    PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of node instance list.");
    return NULL;
  }

  return node_list;

}

/* QD_Element FUNCTION get_coords */
static PyObject *
QD_Element_get_coords(QD_Element* self, PyObject *args, PyObject *kwds){

  if(self->element == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
    return NULL;
  }

  int iTimestep = 0;
  static char *kwlist[] = {"iTimestep",NULL}; // TODO Deprecated!

  if (! PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &iTimestep)){
     return NULL;
  }
    /*
  if (!PyArg_ParseTuple(args, "|i", &iTimestep))
    return NULL;*/

  try{
    return (PyObject*) vector_to_nparray(self->element->get_coords(iTimestep));
  } catch (string e){
    PyErr_SetString(PyExc_RuntimeError, e.c_str());
    return NULL;
  }

}

/* QD_Element FUNCTION get_history */
static PyObject *
QD_Element_get_history(QD_Element* self){

  if(self->element == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
    return NULL;
  }

  return (PyObject*) vector_to_nparray(self->element->get_history_vars());

}

/* QD_Element FUNCTION get_estimated_size */
static PyObject *
QD_Element_get_estimated_size(QD_Element* self){

  if(self->element == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
    return NULL;
  }

  return Py_BuildValue("f",self->element->get_estimated_element_size());

}


/* QD_Element FUNCTION get_type */
static PyObject *
QD_Element_get_type(QD_Element* self){

  if(self->element == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
    return NULL;
  }

  ElementType type = self->element->get_elementType();
  if(type == SHELL){
    return Py_BuildValue("s","shell");
  } else if(type == SOLID){
    return Py_BuildValue("s","solid");
  } else if(type == BEAM) {
    return Py_BuildValue("s","beam");
  } else {
    PyErr_SetString(PyExc_AttributeError,"Unknown element type detected.");
    return NULL;
  }

}
