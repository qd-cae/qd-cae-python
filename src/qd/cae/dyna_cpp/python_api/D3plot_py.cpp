
/* QD_D3plot DEALLOC */
static void
QD_D3plot_dealloc(QD_D3plot* self)
{

  if(self->d3plot != NULL){
    delete self->d3plot;
    self->d3plot = NULL;
    self->femfile.instance = NULL; // = self->d3plot
  }

 #ifdef QD_DEBUG
 cout << "D3plot destroyed" << endl;
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
    self->femfile.instance = NULL;
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
  if(qd::isPyStr(read_states_py)){

    char* variable_c = qd::PyStr2char(read_states_py);
    string variable = string(variable_c);

    variables.push_back(variable);

  } else if(PyList_Check(read_states_py)){

    for(unsigned int ii=0; ii<PySequence_Size(read_states_py); ii++){

        PyObject* item = PyList_GET_ITEM(read_states_py, ii);

        // Check
        if(!qd::isPyStr(item)){
          string message = "Item in list is not of type string.";
          PyErr_SetString(PyExc_SyntaxError,message.c_str() );
          return -1;
        }

        // here we go
        variables.push_back(qd::PyStr2char(item));

    }

  } else {
    // nothing
  }

  // Check if filepath parsing worked
  if(filepath_c){

    try{
      #ifdef QD_MEASURE_TIME
      double wall_time = qd::get_wall_time();
      #endif

      self->d3plot = new D3plot(string(filepath_c), variables, !(useFemzip == 0) );

      #ifdef QD_MEASURE_TIME
      cout << "Wall Time: " << wall_time << endl;
      #endif

      self->femfile.instance = self->d3plot;
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
      PyErr_SetString(PyExc_RuntimeError, "Developer Error d3plot pointer NULL.");
      return NULL;
  }

  return (PyObject*) vector_to_nparray(self->d3plot->get_timesteps());

}

/* FUNCTION info */
static PyObject *
QD_D3plot_info(QD_D3plot* self){

  if (self->d3plot == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "Developer Error d3plot pointer NULL.");
      return NULL;
  }

  self->d3plot->info();

  return Py_None;

}


/* QD_D3plot FUNCTION read_states */
static PyObject *
QD_D3plot_read_states(QD_D3plot* self, PyObject* args){

  if (self->d3plot == NULL) {
      PyErr_SetString(PyExc_RuntimeError, "Developer Error d3plot pointer NULL.");
      return NULL;
  }

  PyObject* argument;
  if (!PyArg_ParseTuple(args, "O", &argument))
    return NULL;

  if(qd::isPyStr(argument)){

    char* variable_c = qd::PyStr2char(argument);
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
        if(!qd::isPyStr(item)){
          string message = "Item in list is not of type string.";
          PyErr_SetString(PyExc_SyntaxError,message.c_str() );
          return NULL;
        }

        // here we go
        variables.push_back(qd::PyStr2char(item));

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
