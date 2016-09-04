

/* DEALLOC */
static void
QD_FEMFile_dealloc(QD_FEMFile* self)
{
   if(self->instance != NULL){
      delete self->instance;
      self->instance = NULL;
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
  if (self != NULL){
    self->instance = NULL;
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

  if (self->instance == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is NULL.");
      return NULL;
  }

  return Py_BuildValue("s",self->instance->get_filepath().c_str());

}


/* FUNCTION get_nodeByID */
static PyObject *
QD_FEMFile_get_nodeByID(QD_FEMFile* self, PyObject* args){

     if (self->instance == NULL) {
       PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is NULL.");
       return NULL;
     }

     PyObject* argument;
     if (!PyArg_ParseTuple(args, "O", &argument))
       return NULL;

     // argument is only one id
     if(PyInt_Check(argument)){

       int nodeID;
       if (!PyArg_ParseTuple(args, "i", &nodeID))
         return NULL;


       if(nodeID < 0){
         PyErr_SetString(PyExc_SyntaxError, "Error, nodeID may not be negative.");
         return NULL;
       }

       PyObject *argList2 = Py_BuildValue("Oi",self ,nodeID);
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
           PyErr_SetString(PyExc_AttributeError,e.c_str());
           Py_DECREF(node_list);
           return NULL;
         }

         if(nodeID < 0){
           Py_DECREF(node_list);
           PyErr_SetString(PyExc_AttributeError, "Error, nodeID may not be negative.");
           return NULL;
         }

         PyObject *argList2 = Py_BuildValue("Oi",self ,nodeID);
         PyObject* ret = PyObject_CallObject((PyObject *) &QD_Node_Type, argList2);
         Py_DECREF(argList2);

         if(ret == NULL){
           Py_DECREF(node_list);
           return NULL;
         }

         check += PyList_SetItem(node_list, ii, ret);

       }

       if(check != 0){
         PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of node list.");
         Py_DECREF(node_list);
         return NULL;
       }

       return node_list;

     }

     PyErr_SetString(PyExc_SyntaxError, "Error, argument is neither int nor list of int.");
     return NULL;


}


/* FUNCTION get_elementByID */
static PyObject *
QD_FEMFile_get_elementByID(QD_FEMFile* self, PyObject* args){

     if (self->instance == NULL) {
       PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is NULL.");
       return NULL;
     }

     PyObject* argument;
     PyObject* elementType_pyobj;
     if (!PyArg_ParseTuple(args, "OO", &elementType_pyobj, &argument))
       return NULL;

     // argument is only one id
     if(PyInt_Check(argument)){

       int elementID;
       try {
         elementID = convert_obj_to_int(argument);
       } catch(string& e) {
         PyErr_SetString(PyExc_AttributeError,e.c_str());
         return NULL;
       }

       if(elementID < 0){
         PyErr_SetString(PyExc_SyntaxError, "Error, elementID may not be negative.");
         return NULL;
       }

       PyObject *argList2 = Py_BuildValue("OOi",self , elementType_pyobj, elementID);
       PyObject* ret = PyObject_CallObject((PyObject *) &QD_Element_Type, argList2);
       Py_DECREF(argList2);

       return ret;

     } else if(PyList_Check(argument)){

       int check=0;
       PyObject* elem_list = PyList_New(PySequence_Size(argument));

       for(unsigned int ii=0; ii<PySequence_Size(argument); ii++){

         PyObject* item = PyList_GET_ITEM(argument, ii);

         int elementID;
         try {
           elementID = convert_obj_to_int(item);
         } catch(string& e) {
           PyErr_SetString(PyExc_AttributeError,e.c_str());
           Py_DECREF(elem_list);
           return NULL;
         }

         if(elementID < 0){
           Py_DECREF(elem_list);
           PyErr_SetString(PyExc_SyntaxError, "Error, elementID may not be negative.");
           return NULL;
         }

         PyObject *argList2 = Py_BuildValue("OOi", self, elementType_pyobj, elementID);
         PyObject* ret = PyObject_CallObject((PyObject *) &QD_Element_Type, argList2);
         Py_DECREF(argList2);

         if(ret == NULL){
           Py_DECREF(elem_list);
           return NULL;
         }

         check += PyList_SetItem(elem_list, ii, ret);

       }

       if(check != 0){
         PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of node list.");
         Py_DECREF(elem_list);
         return NULL;
       }

       return elem_list;

     }

     PyErr_SetString(PyExc_SyntaxError, "Error, argument two is neither int nor list of int.");
     return NULL;

}


/* FUNCTION get_partByID */
static PyObject *
QD_FEMFile_get_partByID(QD_FEMFile* self, PyObject* args){

   if (self->instance == NULL) {
     PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is NULL.");
     return NULL;
   }

   int partID;
   if (!PyArg_ParseTuple(args, "i", &partID))
     return NULL;

   PyObject *argList2 = Py_BuildValue("Oi",self , partID);
   PyObject* ret = PyObject_CallObject((PyObject *) &QD_Part_Type, argList2);
   Py_DECREF(argList2);

   return ret;

}


/* FUNCTION get_parts */
static PyObject *
QD_FEMFile_get_parts(QD_FEMFile* self, PyObject* args){


     if (self->instance == NULL) {
       PyErr_SetString(PyExc_RuntimeError, "Developer Error: pointer to C++ Object is NULL.");
       return NULL;
     }

     // Create list
     PyObject* part_list = PyList_New(self->instance->get_db_parts()->size());

     // fill list
     Part* _part=NULL;
     int check=0;
     for(size_t ii=0; ii < self->instance->get_db_parts()->size(); ++ii){

       _part = self->instance->get_db_parts()->get_part_byIndex(ii+1); // index start at 1

       PyObject *argList2 = Py_BuildValue("Oi",self ,_part->get_partID());
       PyObject* ret = PyObject_CallObject((PyObject *) &QD_Part_Type, argList2);
       Py_DECREF(argList2);

       if(ret == NULL){
         Py_DECREF(part_list);
         PyErr_SetString(PyExc_RuntimeError, "Developer Error during part construction.");
         return NULL;
       }

       check += PyList_SetItem(part_list, ii, ret);
     }

     if(check != 0){
       Py_DECREF(part_list);
       PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of part list.");
       return NULL;
     }

     return part_list;

}
