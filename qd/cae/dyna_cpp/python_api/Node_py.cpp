
/* DEALLOC */
static void
QD_Node_dealloc(QD_Node* self)
{

  Py_DECREF(self->femFile_py);

}


/* NEW */
static PyObject *
QD_Node_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

  QD_Node* self;
  self = (QD_Node *)type->tp_alloc(type, 0);

  // Init vars if any ...
  if (self != nullptr){
    self->node = nullptr;
  }

  return (PyObject*) self;

}


/* INIT */
static int
QD_Node_init(QD_Node *self, PyObject *args, PyObject *kwds)
{

  PyObject* femFile_obj_py;
  PyObject* use_index_py = Py_False;
  QD_FEMFile* femFile_py;
  int iNode;
  bool use_index = false;
  static char *kwlist[] = {const_cast<char*>("femfile"),
                           const_cast<char*>("nodeID"),
                           const_cast<char*>("use_index"), nullptr};

  if (! PyArg_ParseTupleAndKeywords(args, kwds, "Oi|O", kwlist, &femFile_obj_py, &iNode, &use_index_py)){
      return -1;
  }
  use_index = PyObject_IsTrue(use_index_py) != 0;

  if (! PyObject_TypeCheck(femFile_obj_py, &QD_FEMFile_Type)) {
    PyErr_SetString(PyExc_ValueError, "arg #1 not a D3plot or KeyFile in node constructor");
    return -1;
  }

  femFile_py = (QD_FEMFile*) femFile_obj_py;

  if(femFile_py->instance == nullptr){
    string message("Pointer to C++ File-Object is nullptr.");
    PyErr_SetString(PyExc_RuntimeError, message.c_str());
    return -1;
  }

  self->femFile_py = femFile_py;
  Py_INCREF(self->femFile_py);

  if(use_index != 0){
    self->node = femFile_py->instance->get_db_nodes()->get_nodeByIndex(iNode);
  } else {
    self->node = femFile_py->instance->get_db_nodes()->get_nodeByID(iNode);
  }

  if(self->node == nullptr){
    string message("Could not find any node with ID/Index "+to_string(iNode)+".");
    PyErr_SetString(PyExc_RuntimeError,message.c_str());
    Py_DECREF(self->femFile_py);
    return -1;
  }

  return 0;
}

/* FUNCTION richcompare */
static PyObject* 
QD_Node_richcompare(QD_Node *self, PyObject *other, int op){

  PyObject *result = nullptr;

  if( !PyObject_TypeCheck(other, &QD_Node_Type) ){
    PyErr_SetString(PyExc_ValueError, "Comparison of nodes work only with other nodes.");
    return nullptr;
  }

  QD_Node *other_node = (QD_Node*) other;

  switch(op){
      case Py_LT:
          if( self->node->get_nodeID() < other_node->node->get_nodeID() ) { 
            result = Py_True; 
          } else {
            result = Py_False;
          }
          break;
      case Py_LE:
          if( self->node->get_nodeID() <= other_node->node->get_nodeID() ) { 
            result = Py_True; 
          } else {
            result = Py_False;
          }
          break;
      case Py_EQ:
          if( self->node->get_nodeID() == other_node->node->get_nodeID() ) { 
            result = Py_True; 
          } else {
            result = Py_False;
          }
          break;
      case Py_NE:
          if( self->node->get_nodeID() != other_node->node->get_nodeID() ) { 
            result = Py_True; 
          } else {
            result = Py_False;
          }
          break;
      case Py_GT:
          if( self->node->get_nodeID() > other_node->node->get_nodeID() ) { 
            result = Py_True; 
          } else {
            result = Py_False;
          }
          break;
      case Py_GE:
          if( self->node->get_nodeID() >= other_node->node->get_nodeID() ) { 
            result = Py_True; 
          } else {
            result = Py_False;
          }
          break;
  }
  
  Py_XINCREF(result);
  return result;

}


/* FUNCTION get_NodeID */
static PyObject *
QD_Node_get_NodeID(QD_Node* self){

  if(self->node == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to node is nullptr.");
    return nullptr;
  }

  int nodeID = self->node->get_nodeID();

  return Py_BuildValue("i",nodeID);

}


/* FUNCTION get_coords */
static PyObject *
QD_Node_get_coords(QD_Node* self, PyObject *args, PyObject *kwds){

  if(self->node == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to node is nullptr.");
    return nullptr;
  }

  int iTimestep = 0;
  static char *kwlist[] = {const_cast<char*>("iTimestep"),nullptr}; // TODO Deprecated!

  if (! PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &iTimestep)){
      return nullptr;
  }

  try{
    return (PyObject*) vector_to_nparray(self->node->get_coords(iTimestep)); // numpy ... yay
  } catch (string e){
    PyErr_SetString(PyExc_RuntimeError, e.c_str());
    return nullptr;
  }

}

/* FUNCTION get_disp */
static PyObject *
QD_Node_get_disp(QD_Node* self){

  if(self->node == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to node is nullptr.");
    return nullptr;
  }

  // return numpy array
  return (PyObject*) vector_to_nparray(self->node->get_disp()); // numpy ... yay

}


/* FUNCTION get_vel */
static PyObject *
QD_Node_get_vel(QD_Node* self){

  if(self->node == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to node is nullptr.");
    return nullptr;
  }

  return  (PyObject*) vector_to_nparray(self->node->get_vel());

}


/* FUNCTION get_accel */
static PyObject *
QD_Node_get_accel(QD_Node* self){

  if(self->node == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to node is nullptr.");
    return nullptr;
  }

  return (PyObject*) vector_to_nparray(self->node->get_accel());

}


/* FUNCTION get_elements */
static PyObject *
QD_Node_get_elements(QD_Node* self){

  if(self->node == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to element is nullptr.");
    return nullptr;
  }

  vector<Element*> elements = self->node->get_elements();

  int check=0;
  PyObject* element_list = PyList_New(elements.size());
  unsigned int ii=0;

  for(const auto& element : elements){ 

    PyObject* elementType_py;
    if(element->get_elementType() == 1){
      elementType_py = Py_BuildValue("s","beam");
    } else if(element->get_elementType() == 2) {
      elementType_py = Py_BuildValue("s","shell");
    } else if(element->get_elementType() == 3) {
      elementType_py = Py_BuildValue("s","solid");
    } else {
      PyErr_SetString(PyExc_ValueError, "Developer Error, unknown element-type.");
      return nullptr;
    }

    PyObject *argList2 = Py_BuildValue("OOi",self->femFile_py, elementType_py, element->get_elementID());
    PyObject* ret = PyObject_CallObject((PyObject *) &QD_Element_Type, argList2);
    Py_DECREF(argList2);
    Py_DECREF(elementType_py);

    check += PyList_SetItem(element_list, ii, ret);

    ii++;
  }

  if(check != 0){
    PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of element instance list.");
    Py_DECREF(element_list);
    return nullptr;
  }

  return element_list;

}
