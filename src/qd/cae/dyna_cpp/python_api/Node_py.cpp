
/* QD_Node DEALLOC */
static void
QD_Node_dealloc(QD_Node* self)
{

  Py_DECREF(self->d3plot_py);

}


/* QD_Node NEW */
static PyObject *
QD_Node_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

  QD_Node* self;
  self = (QD_Node *)type->tp_alloc(type, 0);

  // Init vars if any ...
  if (self != NULL){
    self->node = NULL;
  }


  return (PyObject*) self;

}


/* QD_Node INIT */
static int
QD_Node_init(QD_Node *self, PyObject *args, PyObject *kwds)
{

  PyObject* d3plot_obj_py;
  QD_D3plot* d3plot_py;
  int nodeID;
  static char *kwlist[] = {"d3plot","nodeID", NULL}; // TODO Deprecated!

  if (! PyArg_ParseTupleAndKeywords(args, kwds, "Oi", kwlist, &d3plot_obj_py, &nodeID)){
      return -1;
  }

  if (! PyObject_TypeCheck(d3plot_obj_py, &QD_D3plot_Type)) {
    PyErr_SetString(PyExc_SyntaxError, "arg #1 not a d3plot in node constructor");
    return -1;
  }

  d3plot_py = (QD_D3plot*) d3plot_obj_py;

  if(d3plot_py->d3plot == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to d3plot-object is NULL.");
    return -1;
  }

  self->d3plot_py = d3plot_py;
  Py_INCREF(self->d3plot_py);
  self->node = d3plot_py->d3plot->get_db_nodes()->get_nodeByID(nodeID);

  if(self->node == NULL){
    string message("Could not find any node with ID "+to_string(nodeID)+".");
    PyErr_SetString(PyExc_RuntimeError,message.c_str());
    return -1;
  }

  return 0;
}

/* QD_Node FUNCTION get_NodeID */
static PyObject *
QD_Node_get_NodeID(QD_Node* self){

  if(self->node == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to node is NULL.");
    return NULL;
  }

  int nodeID = self->node->get_nodeID();

  return Py_BuildValue("i",nodeID);

}


/* QD_Node FUNCTION get_coords */
static PyObject *
QD_Node_get_coords(QD_Node* self, PyObject *args, PyObject *kwds){

  if(self->node == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to node is NULL.");
    return NULL;
  }

  int iTimestep = 0;
  static char *kwlist[] = {"iTimestep",NULL}; // TODO Deprecated!

  if (! PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &iTimestep)){
      return NULL;
  }

  vector<float> coords;
  try{
    coords = self->node->get_coords(iTimestep);
  } catch (const char* e){
    PyErr_SetString(PyExc_RuntimeError, e);
    return NULL;
  } catch (string e){
    PyErr_SetString(PyExc_RuntimeError, e.c_str());
    return NULL;
  }

  int check = 0;
  PyObject* coords_list = PyList_New(coords.size());
  for(unsigned int ii=0; ii<coords.size(); ii++){
    check += PyList_SetItem(coords_list, ii,Py_BuildValue("f",coords[ii]));
  }

  if(check != 0){
    Py_DECREF(coords_list);
    PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of coords list.");
    return NULL;
  }

  return coords_list;

}

/* QD_Node FUNCTION get_disp */
static PyObject *
QD_Node_get_disp(QD_Node* self){

  if(self->node == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to node is NULL.");
    return NULL;
  }

  vector< vector<float> > disp = self->node->get_disp();

  int check0 = 0;
  int check1 = 0;
  PyObject* disp_time_list = PyList_New(disp.size());

  for(unsigned int ii=0; ii<disp.size(); ii++){

    PyObject* disp_list = PyList_New(disp[ii].size());

    for(unsigned int jj=0; jj<disp[ii].size(); jj++){
      check1 += PyList_SetItem(disp_list, jj,Py_BuildValue("f",disp[ii][jj]));
    }

    check0 += PyList_SetItem(disp_time_list, ii, disp_list);
  }

  if( (check0 != 0) | (check1 != 0) ){
    /*
    for (int ii = 0; ii < PyList_Size(disp_time_list); ii++){
      PyObject* disp_list = PyList_GetItem(disp_time_list, ii);
      Py_DECREF(disp_list);
    }
    */
    Py_DECREF(disp_time_list); // TODO: What about the lists in the list?
    PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of coords list.");
    return NULL;
  }

  return disp_time_list;

}


/* QD_Node FUNCTION get_vel */
static PyObject *
QD_Node_get_vel(QD_Node* self){

  if(self->node == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to node is NULL.");
    return NULL;
  }

  vector< vector<float> > vel = self->node->get_vel();

  int check0 = 0;
  int check1 = 0;
  PyObject* vel_time_list = PyList_New(vel.size());

  for(unsigned int ii=0; ii<vel.size(); ii++){

    PyObject* vel_list = PyList_New(vel[ii].size());

    for(unsigned int jj=0; jj<vel[ii].size(); jj++){
      check1 += PyList_SetItem(vel_list, jj,Py_BuildValue("f",vel[ii][jj]));
    }

    check0 += PyList_SetItem(vel_time_list, ii, vel_list);
  }

  if( (check0 != 0) | (check1 != 0) ){
    /*
    for (int ii = 0; ii < PyList_Size(disp_time_list); ii++){
      PyObject* disp_list = PyList_GetItem(disp_time_list, ii);
      Py_DECREF(disp_list);
    }
    */
    Py_DECREF(vel_time_list); // TODO: What about the lists in the list?
    PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of velocity list.");
    return NULL;
  }

  return vel_time_list;

}


/* QD_Node FUNCTION get_accel */
static PyObject *
QD_Node_get_accel(QD_Node* self){

  if(self->node == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to node is NULL.");
    return NULL;
  }

  vector< vector<float> > accel = self->node->get_accel();

  int check0 = 0;
  int check1 = 0;
  PyObject* accel_time_list = PyList_New(accel.size());

  for(unsigned int ii=0; ii<accel.size(); ii++){

    PyObject* accel_list = PyList_New(accel[ii].size());

    for(unsigned int jj=0; jj<accel[ii].size(); jj++){
      check1 += PyList_SetItem(accel_list, jj,Py_BuildValue("f",accel[ii][jj]));
    }

    check0 += PyList_SetItem(accel_time_list, ii, accel_list);
  }

  if( (check0 != 0) | (check1 != 0) ){
    /*
    for (int ii = 0; ii < PyList_Size(disp_time_list); ii++){
      PyObject* disp_list = PyList_GetItem(disp_time_list, ii);
      Py_DECREF(disp_list);
    }
    */
    Py_DECREF(accel_time_list); // TODO: What about the lists in the list?
    PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of velocity list.");
    return NULL;
  }

  return accel_time_list;

}

/* QD_Node FUNCTION get_elements */
static PyObject *
QD_Node_get_elements(QD_Node* self){

  if(self->node == NULL){
    PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
    return NULL;
  }

  set<Element*> elements = self->node->get_elements();

  int check=0;
  PyObject* element_list = PyList_New(elements.size());
  unsigned int ii=0;

  for(set<Element*>::iterator it=elements.begin(); it != elements.end(); ++it){
//    for(auto element : elements){ // -std=c++11 rulez
    Element* element = *it;

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
