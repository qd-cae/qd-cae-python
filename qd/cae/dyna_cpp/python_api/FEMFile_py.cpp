

/* DEALLOC */
static void
QD_FEMFile_dealloc(QD_FEMFile* self)
{
   if(self->instance != nullptr){
      delete self->instance;
      self->instance = nullptr;
   }

   #ifdef QD_DEBUG
   cout << "QD_FEMFile Destructor" << endl;
   #endif
}

/* NEW */
static PyObject *
QD_FEMFile_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

  QD_FEMFile* self;
  self = (QD_FEMFile *)type->tp_alloc(type, 0);

  // Init vars if any ...
  if (self != nullptr){
    self->instance = nullptr;
  }

  return (PyObject*) self;

}

/* INIT */
static int
QD_FEMFile_init(QD_FEMFile *self, PyObject *args, PyObject *kwds)
{
  return 0;
}

/*  FUNCTION get_filepath */
static PyObject *
QD_FEMFile_get_filepath(QD_FEMFile* self){

  if (self->instance == nullptr) {
      PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is nullptr.");
      return nullptr;
  }

  return Py_BuildValue("s",self->instance->get_filepath().c_str());

}


/*  FUNCTION get_nNodes */
static PyObject *
QD_FEMFile_get_nNodes(QD_FEMFile* self){

  if (self->instance == nullptr) {
      PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is nullptr.");
      return nullptr;
  }

  return Py_BuildValue("i",self->instance->get_db_nodes()->size());

}


/*  FUNCTION get_nElements */
static PyObject *
QD_FEMFile_get_nElements(QD_FEMFile* self, PyObject* args){

  if (self->instance == nullptr) {
      PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is nullptr.");
      return nullptr;
  }

  // Parse args
  PyObject* arg = Py_None;
  string element_type = "";
  if (!PyArg_ParseTuple(args, "|O", &arg))
    return nullptr;

  if(arg == Py_None){
     //Nothing
  } else if( qd::isPyStr(arg) ){
     element_type = qd::PyStr2char(arg);
  } else {
     PyErr_SetString(PyExc_ValueError,"Argument is not a string.");
     return nullptr;
  }

  if( !element_type.empty() ){
     if( element_type == "shell" ){
       return Py_BuildValue("i",self->instance->get_db_elements()->size(SHELL));
     } else if ( element_type == "solid" ){
       return Py_BuildValue("i",self->instance->get_db_elements()->size(SOLID));
     } else if ( element_type == "beam" ){
       return Py_BuildValue("i",self->instance->get_db_elements()->size(BEAM));
     } else {
       PyErr_SetString(PyExc_ValueError, "Unknown element type, please try beam, shell or solid.");
       return nullptr;
     }
  }

  return Py_BuildValue("i",self->instance->get_db_elements()->size());

}


/* QD_Part FUNCTION get_nodes */
static PyObject*
QD_FEMFile_get_nodes(QD_FEMFile *self){

  if(self->instance == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to C++ object is nullptr.");
    return nullptr;
  }

  DB_Nodes* db_nodes = self->instance->get_db_nodes();

  int check=0;
  PyObject* node_list = PyList_New(db_nodes->size());

  size_t ii=0;
  for(size_t iNode=0; iNode < db_nodes->size(); ++iNode){

    PyObject *argList2 = Py_BuildValue("Oi",self, db_nodes->get_nodeByIndex(iNode)->get_nodeID());
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


/* FUNCTION get_elements */
static PyObject*
QD_FEMFile_get_elements(QD_FEMFile *self, PyObject* args){

  if(self->instance == nullptr){
    PyErr_SetString(PyExc_RuntimeError,"Pointer to C++ object is nullptr.");
    return nullptr;
  }

  // Parse args
  PyObject* arg = Py_None;
  string element_type = "";
  if (!PyArg_ParseTuple(args, "|O", &arg))
    return nullptr;

  if(arg == Py_None){
     //Nothing
  } else if( qd::isPyStr(arg) ){
     element_type = qd::PyStr2char(arg);
  } else {
     PyErr_SetString(PyExc_ValueError,"Argument is not a string.");
     return nullptr;
  }


  ElementType eType = NONE;
  if(element_type == "beam"){
     eType = BEAM;
  } else if (element_type == "shell"){
     eType = SHELL;
  } else if (element_type == "solid"){
     eType = SOLID;
  } else if ( element_type.empty() ) {
     eType = NONE;
  } else {
     PyErr_SetString(PyExc_ValueError,"Unknown element type. Use beam, shell or solid.");
     return nullptr;
  }

  DB_Elements* db_elements = self->instance->get_db_elements();

  // Do the thing
  // Add all types if NONE
  size_t iElementTotal = 0;
  int check=0;
  PyObject* element_list = PyList_New(db_elements->size(eType)); // allocate
  PyObject* ret;
  PyObject *argList2;

  if( (eType == BEAM) || (eType == NONE) ){

     PyObject* elementType_py = Py_BuildValue("s","beam");
     for(size_t iElement=0; iElement<db_elements->size(BEAM); ++iElement){

        argList2 = Py_BuildValue("OOi",self, elementType_py, db_elements->get_elementByIndex(BEAM,iElement)->get_elementID());
        ret = PyObject_CallObject((PyObject *) &QD_Element_Type, argList2);
        check += PyList_SetItem(element_list, iElementTotal, ret);
        Py_DECREF(argList2);
        iElementTotal += 1;
     }
     Py_DECREF(elementType_py);

  }

  if( (eType == SHELL) || (eType == NONE) ){

     PyObject* elementType_py = Py_BuildValue("s","shell");
     for(size_t iElement=0; iElement<db_elements->size(SHELL); ++iElement){

        argList2 = Py_BuildValue("OOi",self, elementType_py, db_elements->get_elementByIndex(SHELL,iElement)->get_elementID());
        ret = PyObject_CallObject((PyObject *) &QD_Element_Type, argList2);
        ret = PyObject_CallObject((PyObject *) &QD_Element_Type, argList2);
        check += PyList_SetItem(element_list, iElementTotal, ret);
        Py_DECREF(argList2);
        iElementTotal += 1;
     }
     Py_DECREF(elementType_py);

  }

  if( (eType == SOLID) || (eType == NONE) ){

     PyObject* elementType_py = Py_BuildValue("s","solid");
     for(size_t iElement=0; iElement<db_elements->size(SOLID); ++iElement){
        argList2 = Py_BuildValue("OOi",self, elementType_py, db_elements->get_elementByIndex(SOLID,iElement)->get_elementID());
        ret = PyObject_CallObject((PyObject *) &QD_Element_Type, argList2);
        check += PyList_SetItem(element_list, iElementTotal, ret);
        Py_DECREF(argList2);
        iElementTotal += 1;
     }
     Py_DECREF(elementType_py);

  }

  if(check != 0){
    PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of element instance list.");
    Py_DECREF(element_list);
    return nullptr;
  }

  return element_list;

}


/* FUNCTION get_nodeByID */
static PyObject *
QD_FEMFile_get_nodeByID(QD_FEMFile* self, PyObject* args){

     if (self->instance == nullptr) {
       PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is nullptr.");
       return nullptr;
     }

     PyObject* argument;
     if (!PyArg_ParseTuple(args, "O", &argument))
       return nullptr;

     // argument is only one id
     if(PyObject_isIntegral(argument)){

       int nodeID;
       if (!PyArg_ParseTuple(args, "i", &nodeID))
         return nullptr;


       if(nodeID < 0){
         PyErr_SetString(PyExc_ValueError, "Error, nodeID may not be negative.");
         return nullptr;
       }

       PyObject *argList2 = Py_BuildValue("OiO", self ,nodeID, Py_False);
       PyObject* ret = PyObject_CallObject((PyObject *) &QD_Node_Type, argList2);
       Py_DECREF(argList2);

       return ret;

     // argument is a list of id's
     } else if(PyList_Check(argument)){

       int check=0;
       PyObject* node_list = PyList_New(PySequence_Size(argument));

       for(long ii=0; ii<PySequence_Size(argument); ii++){

         PyObject* item = PyList_GET_ITEM(argument, ii);

         int nodeID;
         try {
           nodeID = convert_obj_to_int(item);
         } catch(string& e) {
           PyErr_SetString(PyExc_ValueError,e.c_str());
           Py_DECREF(node_list);
           return nullptr;
         }

         if(nodeID < 0){
           Py_DECREF(node_list);
           PyErr_SetString(PyExc_ValueError, "Error, nodeID may not be negative.");
           return nullptr;
         }

         PyObject *argList2 = Py_BuildValue("OiO",self ,nodeID, Py_False);
         try{

           PyObject* ret = PyObject_CallObject((PyObject *) &QD_Node_Type, argList2);
           
           if(ret == nullptr)
             throw(string("Could somehow not create Python Node."));

           check += PyList_SetItem(node_list, ii, ret);

         } catch (const string& message) {
           PyErr_SetString(PyExc_RuntimeError, message.c_str());
           Py_DECREF(argList2);
           Py_DECREF(node_list);
           return nullptr;
         }
         Py_DECREF(argList2);

       }

       if(check != 0){
         PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of node list.");
         Py_DECREF(node_list);
         return nullptr;
       }

       return node_list;

     }

     PyErr_SetString(PyExc_ValueError, "Error, argument is neither int nor list of int.");
     return nullptr;


}


/* FUNCTION get_nodeByIndex */
static PyObject *
QD_FEMFile_get_nodeByIndex(QD_FEMFile* self, PyObject* args){

     if (self->instance == nullptr) {
       PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is nullptr.");
       return nullptr;
     }

     PyObject* argument;
     if (!PyArg_ParseTuple(args, "O", &argument))
       return nullptr;

     // argument is only one id
     if(PyObject_isIntegral(argument)){

       int nodeIndex;
       if (!PyArg_ParseTuple(args, "i", &nodeIndex))
         return nullptr;


       if(nodeIndex < 0){
         PyErr_SetString(PyExc_ValueError, "Error, nodeIndex may not be negative.");
         return nullptr;
       }

       PyObject *argList2 = Py_BuildValue("OiO", self, nodeIndex, Py_True);
       PyObject* ret = PyObject_CallObject((PyObject *) &QD_Node_Type, argList2);
       Py_DECREF(argList2);

       return ret;

     // argument is a list of id's
     } else if(PyList_Check(argument)){

       int check=0;
       PyObject* node_list = PyList_New(PySequence_Size(argument));

       for(unsigned int ii=0; ii<PySequence_Size(argument); ii++){

         PyObject* item = PyList_GET_ITEM(argument, ii);

         int nodeID;
         try {
           nodeID = convert_obj_to_int(item);
         } catch(string& e) {
           PyErr_SetString(PyExc_ValueError,e.c_str());
           Py_DECREF(node_list);
           return nullptr;
         }

         if(nodeID < 0){
           Py_DECREF(node_list);
           PyErr_SetString(PyExc_ValueError, "Error, nodeID may not be negative.");
           return nullptr;
         }

         PyObject *argList2 = Py_BuildValue("OiO",self ,nodeID, Py_True);
         PyObject* ret = PyObject_CallObject((PyObject *) &QD_Node_Type, argList2);
         Py_DECREF(argList2);

         if(ret == nullptr){
           Py_DECREF(node_list);
           return nullptr;
         }

         check += PyList_SetItem(node_list, ii, ret);

       }

       if(check != 0){
         PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of node list.");
         Py_DECREF(node_list);
         return nullptr;
       }

       return node_list;

     }

     PyErr_SetString(PyExc_ValueError, "Error, argument is neither int nor list of int.");
     return nullptr;


}


/* FUNCTION get_elementByID */
static PyObject *
QD_FEMFile_get_elementByID(QD_FEMFile* self, PyObject* args){

     if (self->instance == nullptr) {
       PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is nullptr.");
       return nullptr;
     }

     PyObject* argument;
     PyObject* elementType_pyobj;
     if (!PyArg_ParseTuple(args, "OO", &elementType_pyobj, &argument))
       return nullptr;

     // argument is only one id
     if(PyObject_isIntegral(argument)){

       int elementID;
       try {
         elementID = convert_obj_to_int(argument);
       } catch(string& e) {
         PyErr_SetString(PyExc_ValueError,e.c_str());
         return nullptr;
       }

       if(elementID < 0){
         PyErr_SetString(PyExc_ValueError, "Error, elementID may not be negative.");
         return nullptr;
       }

       PyObject *argList2 = Py_BuildValue("OOi",self , elementType_pyobj, elementID);
       PyObject* ret = PyObject_CallObject((PyObject *) &QD_Element_Type, argList2);
       Py_DECREF(argList2);

       return ret;

     } else if(PyList_Check(argument)){

       int check=0;
       PyObject* elem_list = PyList_New(PySequence_Size(argument));

       for(long ii=0; ii<PySequence_Size(argument); ii++){

         PyObject* item = PyList_GET_ITEM(argument, ii);

         int elementID;
         try {
           elementID = convert_obj_to_int(item);
         } catch(string& e) {
           PyErr_SetString(PyExc_ValueError,e.c_str());
           Py_DECREF(elem_list);
           return nullptr;
         }

         if(elementID < 0){
           Py_DECREF(elem_list);
           PyErr_SetString(PyExc_ValueError, "Error, elementID may not be negative.");
           return nullptr;
         }

         PyObject *argList2 = Py_BuildValue("OOi", self, elementType_pyobj, elementID);
         PyObject* ret = PyObject_CallObject((PyObject *) &QD_Element_Type, argList2);
         Py_DECREF(argList2);

         if(ret == nullptr){
           Py_DECREF(elem_list);
           return nullptr;
         }

         check += PyList_SetItem(elem_list, ii, ret);

       }

       if(check != 0){
         PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of node list.");
         Py_DECREF(elem_list);
         return nullptr;
       }

       return elem_list;

     }

     PyErr_SetString(PyExc_ValueError, "Error, argument two is neither int nor list of int.");
     return nullptr;

}


/* FUNCTION get_partByID */
static PyObject *
QD_FEMFile_get_partByID(QD_FEMFile* self, PyObject* args){

   if (self->instance == nullptr) {
     PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is nullptr.");
     return nullptr;
   }

   int partID;
   if (!PyArg_ParseTuple(args, "i", &partID))
     return nullptr;

   PyObject *argList2 = Py_BuildValue("Oi",self , partID);
   PyObject* ret = PyObject_CallObject((PyObject *) &QD_Part_Type, argList2);
   Py_DECREF(argList2);

   return ret;

}


/* FUNCTION get_mesh */
static PyObject *
QD_FEMFile_get_mesh(QD_FEMFile* self, PyObject* args){

   if (self->instance == nullptr) {
     PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is nullptr.");
     return nullptr;
   }

   /*
   int iTimestep = 0;
   if (!PyArg_ParseTuple(args, "|i", &iTimestep))
     return nullptr;

   // Databases
   DB_Nodes* db_nodes = self->instance->get_db_nodes();
   DB_Elements* db_elements = self->instance->get_db_elements();

   // Return values
   PyObject* nodes_array = Py_None;
   PyObject* elements_array = Py_None;

   // Extract nodes
   size_t nNodes = db_nodes->size();
   map<int,int> node_ids2index;
   if(nNodes != 0){

      Node* current_node = nullptr;
      vector< vector<float> > node_coords(nNodes);
      for(size_t iNode=0; iNode < nNodes; ++iNode){
         current_node = db_nodes->get_nodeByIndex(iNode);
         node_coords[iNode] = current_node->get_coords(iTimestep);
         node_ids2index.insert(pair<int,size_t>(current_node->get_nodeID(),iNode));
      }

      nodes_array = (PyObject*) vector_to_nparray(node_coords);
   }

   // Extract elements
   size_t nElements = db_elements->size();
   if (nElements != 0){

      size_t nElements2 = db_elements->size(BEAM);
      size_t nElements4 = db_elements->size(SHELL);
      size_t nElements8 = db_elements->size(SOLID);

      size_t iNode = 0;
      vector<int>::iterator it;
      vector<int> node_ids;
      vector<int> node_indexes;
      vector< vector<int> > element4_trias;

      for(size_t iElement=0; iElement < nElements4; ++iElement){

         node_ids = db_elements->get_elementByIndex(SHELL,iElement)->get_node_ids();
         node_indexes = vector<int>(node_ids.size());

         iNode = 0;
         //for(size_t iNode=0; iNode < node_ids.size(); ++iNode){
         for(it = node_ids.begin(); it != node_ids.end(); ++it){
            node_indexes[iNode] = node_ids2index[*it];
            ++iNode;
         }
         if(node_ids.size() == 3){
            element4_trias.push_back(node_indexes);
         }

      }

      elements_array = (PyObject*) vector_to_nparray(element4_trias,NPY_INT32);

   }

   // return Tuple
   return Py_BuildValue("[O,O]",nodes_array,elements_array);
   */

   return Py_None;
   
}


/* FUNCTION get_parts */
static PyObject *
QD_FEMFile_get_parts(QD_FEMFile* self){


     if (self->instance == nullptr) {
       PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is nullptr.");
       return nullptr;
     }

     // Create list
     PyObject* part_list = PyList_New(self->instance->get_db_parts()->size());

     // fill list
     Part* _part=nullptr;
     int check=0;
     for(size_t ii=0; ii < self->instance->get_db_parts()->size(); ++ii){

       _part = self->instance->get_db_parts()->get_part_byIndex(ii);
       if(_part == nullptr){
         PyErr_SetString(PyExc_RuntimeError, "Developer error, an nullptr was returned instead of a part.");
       }

       PyObject *argList2 = Py_BuildValue("Oi",self ,_part->get_partID());
       PyObject* ret = PyObject_CallObject((PyObject *) &QD_Part_Type, argList2);
       Py_DECREF(argList2);

       if(ret == nullptr){
         Py_DECREF(part_list);
         PyErr_SetString(PyExc_RuntimeError, "Developer Error during part construction.");
         return nullptr;
       }

       check += PyList_SetItem(part_list, ii, ret);
     }

     if(check != 0){
       Py_DECREF(part_list);
       PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of part list.");
       return nullptr;
     }

     return part_list;

}
