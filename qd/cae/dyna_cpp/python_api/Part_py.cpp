

/* QD_Part DEALLOC */
static void
QD_Part_dealloc(QD_Part* self){
  Py_DECREF(self->femFile_py);
}


/* QD_Part NEW */
static PyObject *
QD_Part_new(PyTypeObject *type, PyObject *args, PyObject *kwds){

  QD_Part* self;
  self = (QD_Part *)type->tp_alloc(type, 0);

  // Init vars if any ...
  if (self != nullptr){
    self->part = nullptr;
  }


  return (PyObject*) self;

}

/* QD_Part INIT */
static int
QD_Part_init(QD_Part *self, PyObject *args, PyObject *kwds){

  PyObject* femFile_obj_py;
  int partID;
  static char *kwlist[] = {const_cast<char*>("femfile"),
                           const_cast<char*>("part_id"), nullptr};


  if (! PyArg_ParseTupleAndKeywords(args, kwds, "Oi", kwlist, &femFile_obj_py, &partID)){
      return -1;
  }

  if (! PyObject_TypeCheck(femFile_obj_py, &QD_FEMFile_Type)) {
    PyErr_SetString(PyExc_ValueError, "arg #1 not a D3plot or KeyFile in part constructor");
    return -1;
  }
  QD_FEMFile* femFile_py = (QD_FEMFile*) femFile_obj_py;

  if(femFile_py->instance == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to C++ File-Object is nullptr.");
    return -1;
  }

  self->femFile_py = femFile_py;
  Py_INCREF(self->femFile_py);
  self->part = femFile_py->instance->get_db_parts()->get_part_byID(partID);

  if(self->part == nullptr){
    PyErr_SetString(PyExc_RuntimeError,string("Could not find any part with ID: "+to_string(partID)+".").c_str());
    return -1;
  }

  return 0;

}


/* FUNCTION richcompare */
static PyObject * 
QD_Part_richcompare(QD_Part *self, PyObject *other, int op){

  PyObject *result = nullptr;

  if( !PyObject_TypeCheck(other, &QD_Part_Type) ){
    PyErr_SetString(PyExc_ValueError, "Comparison of parts work only with other parts.");
    return nullptr;
  }

  QD_Part *other_cpp = (QD_Part*) other;

  switch(op){
      case Py_LT:
          if( self->part->get_partID() < other_cpp->part->get_partID() ) { 
            result = Py_True; 
          } else {
            result = Py_False;
          }
          break;
      case Py_LE:
          if( self->part->get_partID() <= other_cpp->part->get_partID() ) { 
            result = Py_True; 
          } else {
            result = Py_False;
          }
          break;
      case Py_EQ:
          if( self->part->get_partID() == other_cpp->part->get_partID() ) { 
            result = Py_True; 
          } else {
            result = Py_False;
          }
          break;
      case Py_NE:
          if( self->part->get_partID() != other_cpp->part->get_partID() ) { 
            result = Py_True; 
          } else {
            result = Py_False;
          }
          break;
      case Py_GT:
          if( self->part->get_partID() > other_cpp->part->get_partID() ) { 
            result = Py_True; 
          } else {
            result = Py_False;
          }
          break;
      case Py_GE:
          if( self->part->get_partID() >= other_cpp->part->get_partID() ) { 
            result = Py_True; 
          } else {
            result = Py_False;
          }
          break;
  }
  
  Py_XINCREF(result);
  return result;

}


static PyObject*
QD_Part_get_id(QD_Part *self){

  if(self->part == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to part is nullptr.");
    return nullptr;
  }

  return Py_BuildValue("i",self->part->get_partID());

}


/* QD_Part FUNCTION get_name */
static PyObject*
QD_Part_get_name(QD_Part *self){

  if(self->part == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to part is nullptr.");
    return nullptr;
  }

  string partName = self->part->get_name();

  return Py_BuildValue("s",partName.c_str());

}


/* QD_Part FUNCTION get_nodes */
static PyObject*
QD_Part_get_nodes(QD_Part *self){

  if(self->part == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to part is nullptr.");
    return nullptr;
  }

  vector<Node*> nodes = self->part->get_nodes();

  int check=0;
  PyObject* node_list = PyList_New(nodes.size());

  size_t ii=0;
//   for(auto node : nodes){ // -std=c++11 rulez
  Node* node = nullptr;
  for(vector<Node*>::iterator it=nodes.begin(); it != nodes.end(); it++){
    node = *it;

    PyObject *argList2 = Py_BuildValue("Oi",self->femFile_py ,node->get_nodeID());
    PyObject* ret = PyObject_CallObject((PyObject *) &QD_Node_Type, argList2);
    Py_DECREF(argList2);

    check += PyList_SetItem(node_list, ii, ret);

    ii++;
  }

  if(check != 0){
    PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of node instance list.");
    Py_DECREF(node_list);
    return nullptr;
  }

  return node_list;

}


/* QD_Part FUNCTION get_elements */
static PyObject*
QD_Part_get_elements(QD_Part *self, PyObject *args){

  if(self->part == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to part is nullptr.");
    return nullptr;
  }

  // Parse args
  PyObject* _filter_etype_py = Py_None;
  if (!PyArg_ParseTuple(args, "|O", &_filter_etype_py))
    return nullptr;  

  // check filter type
  ElementType _filter_etype = NONE;
  if( _filter_etype_py != Py_None ){
    if( qd::isPyStr(_filter_etype_py) ){
      
      string _filter_etype_str = string( qd::PyStr2char(_filter_etype_py) );
      if( _filter_etype_str == "shell" ){
        _filter_etype = SHELL;
      } else if ( _filter_etype_str == "solid" ){
        _filter_etype = SOLID;
      } else if ( _filter_etype_str == "beam" ){
        _filter_etype = BEAM;
      } else {
        string err_msg =  "Error: unknown element-type \""
                        + _filter_etype_str
                        + "\", use beam, shell or solid.";
        PyErr_SetString(PyExc_ValueError, err_msg.c_str() );
        return nullptr;
      }

    } else { // argument not str

      string err_msg =  "Error: argument element_type has unknown type, use str or None";
      PyErr_SetString(PyExc_ValueError, err_msg.c_str() );
      return nullptr;

    }
  }

  vector<Element*> elements = self->part->get_elements(_filter_etype);

  int check=0;
  PyObject* element_list = PyList_New(elements.size());

  size_t ii=0;
  Element* element = nullptr;
  //for(vector<Element*>::const_iterator it=elements.begin(); it != elements.end(); it++){
  for( const auto& element : elements){

    // create string for constructor
    PyObject* elementType_py;
    if(element->get_elementType() == BEAM){
      elementType_py = Py_BuildValue("s","beam");
    } else if(element->get_elementType() == SHELL) {
      elementType_py = Py_BuildValue("s","shell");
    } else if(element->get_elementType() == SOLID) {
      elementType_py = Py_BuildValue("s","solid");
    } else {
      PyErr_SetString(PyExc_RuntimeError, "Developer Error, unknown element-type.");
      return nullptr;
    }

    // create python object
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
