
/* QD_D3plot DEALLOC */
static void
QD_D3plot_dealloc(QD_D3plot* self)
{

  if(self->d3plot != NULL){
    delete self->d3plot;
    self->d3plot = NULL;
  }

 #ifdef QD_DEBUG
 cout << "D3plot destructor" << endl;
 #endif

}

/* QD_D3plot NEW */
static PyObject *
QD_D3plot_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

  QD_D3plot* self;
  self = (QD_D3plot *)type->tp_alloc(type, 0);

  // Init vars if any ...
  if (self != NULL){
    self->d3plot = NULL;
    self->femfile.femfile_ptr = NULL;
  }

  return (PyObject*) self;

}


/* QD_D3plot INIT */
static int
QD_D3plot_init(QD_D3plot *self, PyObject *args, PyObject *kwds)
{

  int useFemzip = 0;
  char* filepath_c;
  static char *kwlist[] = {"filepath","use_femzip","read_states",NULL}; // TODO Deprecated!


  PyObject* read_states_py = Py_None;
  if (! PyArg_ParseTupleAndKeywords(args, kwds, "s|bO", kwlist, &filepath_c,&useFemzip,&read_states_py)){
      return -1;
  }

  vector<string> variables;
  if(PyString_Check(read_states_py)){

    char* variable_c = PyString_AsString(read_states_py);
    string variable = string(variable_c);

    variables.push_back(variable);

  } else if(PyList_Check(read_states_py)){

    for(unsigned int ii=0; ii<PySequence_Size(read_states_py); ii++){

        PyObject* item = PyList_GET_ITEM(read_states_py, ii);

        // Check
        if(!PyString_Check(item)){
          string message = "Item in list is not of type string.";
          PyErr_SetString(PyExc_SyntaxError,message.c_str() );
          return -1;
        }

        // here we go
        variables.push_back(PyString_AsString(item));

    }

  } else {
    // nothing
  }

  // Check if filepath parsing worked
  if(filepath_c){

    try{
      self->d3plot = new D3plot(string(filepath_c), variables, (bool) useFemzip);
      self->femfile.femfile_ptr = self->d3plot;
    } catch (const char* e){
      PyErr_SetString(PyExc_RuntimeError, e);
      return -1;
    } catch (string e){
      PyErr_SetString(PyExc_RuntimeError, e.c_str());
      return -1;
    }

    //self->d3plot = new D3plot(string(filepath_c));
  } else {
    PyErr_SetString(PyExc_SyntaxError,"Filepath is NULL");
    return -1;
  }

  return 0;
}


/* QD_D3plot FUNCTION get_timesteps */
static PyObject *
QD_D3plot_get_timesteps(QD_D3plot* self){

  if (self->d3plot == NULL) {
      PyErr_SetString(PyExc_AttributeError, "Developer Error d3plot pointer NULL.");
      return NULL;
  }

  vector<float> timesteps = self->d3plot->get_timesteps();

  int check = 0;
  PyObject* timesteps_list = PyList_New(timesteps.size());
  for(unsigned int ii=0; ii<timesteps.size(); ii++){
    check += PyList_SetItem(timesteps_list, ii,Py_BuildValue("f",timesteps[ii]));
  }

  if(check != 0){
    Py_DECREF(timesteps_list);
    PyErr_SetString(PyExc_AttributeError, "Developer Error during build of timestep list.");
    return NULL;
  }

  return timesteps_list;

}

/* QD_D3plot FUNCTION get_filepath */
static PyObject *
QD_D3plot_get_filepath(QD_D3plot* self){

  if (self->d3plot == NULL) {
      PyErr_SetString(PyExc_AttributeError, "Developer Error d3plot pointer NULL.");
      return NULL;
  }

  return Py_BuildValue("s",self->d3plot->get_filepath().c_str());

}

/* QD_D3plot FUNCTION read_states */
static PyObject *
QD_D3plot_read_states(QD_D3plot* self, PyObject* args){

  if (self->d3plot == NULL) {
      PyErr_SetString(PyExc_AttributeError, "Developer Error d3plot pointer NULL.");
      return NULL;
  }

  PyObject* argument;
  if (!PyArg_ParseTuple(args, "O", &argument))
    return NULL;

  if(PyString_Check(argument)){

    char* variable_c = PyString_AsString(argument);
    string variable = string(variable_c);

    vector<string> variables;
    variables.push_back(variable);

    try{
      self->d3plot->read_states(variables);
    } catch (const char* e){
      PyErr_SetString(PyExc_RuntimeError, e);
      return NULL;
    } catch (string e){
      PyErr_SetString(PyExc_RuntimeError, e.c_str());
      return NULL;
    }

    return Py_None;

  } else if(PyList_Check(argument)){

      vector<string> variables;
      for(unsigned int ii=0; ii<PySequence_Size(argument); ii++){

        PyObject* item = PyList_GET_ITEM(argument, ii);

        // Check
        if(!PyString_Check(item)){
          string message = "Item in list is not of type string.";
          PyErr_SetString(PyExc_SyntaxError,message.c_str() );
          return NULL;
        }

        // here we go
        variables.push_back(PyString_AsString(item));

      }

      try{
        self->d3plot->read_states(variables);
      } catch (const char* e){
        PyErr_SetString(PyExc_RuntimeError, e);
        return NULL;
      } catch (string e){
        PyErr_SetString(PyExc_RuntimeError, e.c_str());
        return NULL;
      }

      return Py_None;

  }

  PyErr_SetString(PyExc_SyntaxError, "Error, argument is neither int nor list of int.");
  return NULL;

}

/* QD_D3plot FUNCTION get_nodeByID */
static PyObject *
QD_D3plot_get_nodeByID(QD_D3plot* self, PyObject* args){

  if (self->d3plot == NULL) {
    PyErr_SetString(PyExc_AttributeError, "Developer Error d3plot pointer NULL.");
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
      } catch(int e) {
        Py_DECREF(node_list);
        return NULL;
      }

      if(nodeID < 0){
        Py_DECREF(node_list);
        PyErr_SetString(PyExc_SyntaxError, "Error, nodeID may not be negative.");
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


/* QD_D3plot FUNCTION get_elementByID */
static PyObject *
QD_D3plot_get_elementByID(QD_D3plot* self, PyObject* args){

  if (self->d3plot == NULL) {
    PyErr_SetString(PyExc_AttributeError, "Developer Error d3plot pointer NULL.");
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
    } catch(int e) {
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
      } catch(int e) {
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


/* QD_D3plot FUNCTION get_partByID */
static PyObject *
QD_D3plot_get_partByID(QD_D3plot* self, PyObject* args){

  if (self->d3plot == NULL) {
    PyErr_SetString(PyExc_AttributeError, "Developer Error d3plot pointer NULL.");
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

/* QD_D3plot FUNCTION get_parts */
static PyObject *
QD_D3plot_get_parts(QD_D3plot* self, PyObject* args){

  if (self->d3plot == NULL) {
    PyErr_SetString(PyExc_AttributeError, "Developer Error d3plot pointer NULL.");
    return NULL;
  }

  // Create list
  PyObject* part_list = PyList_New(self->d3plot->get_db_parts()->size());

  // fill list
  Part* _part=NULL;
  int check=0;
  for(unsigned int ii=0; ii < self->d3plot->get_db_parts()->size(); ++ii){

    _part = self->d3plot->get_db_parts()->get_part_byIndex(ii+1); // index start at 1

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
