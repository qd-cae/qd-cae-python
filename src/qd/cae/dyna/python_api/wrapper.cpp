
#include <Python.h>
#include "D3plot_py.h"
#include "Node_py.h"
#include "Element_py.h"
#include "Part_py.h"
#include <limits>
#include <string.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <set>
#include "../utility/TextUtility.h"
#include "../dyna/d3plot.h"
#include "../db/DB_Elements.h"
#include "../db/DB_Nodes.h"
#include "../db/DB_Parts.h"
#include "../db/Node.h"
#include "../db/Element.h"

using namespace std;

extern "C" {

  /* convert_obj_to_int: cast obj to long with checks */
  static int
  convert_obj_to_int(PyObject* item){

    if(!PyInt_Check(item)){
          PyErr_SetString(PyExc_SyntaxError, "Error, argument list entry is not an integer.");
          throw(-1);
    }

    long nodeID_long = PyLong_AsLong(item);

    // Overflow cast check
    if((long) std::numeric_limits<int>::max() < nodeID_long){
      PyErr_SetString(PyExc_SyntaxError, "Integer overflow error.");
      throw(-1);
    } else if ((long) std::numeric_limits<int>::min() > nodeID_long){
      PyErr_SetString(PyExc_SyntaxError, "Integer underflow error.");
      throw(-1);
    }

    return (int) PyLong_AsLong(item);

  }



  /*******************************************************/
  /*                                                     */
  /*                  C D - D 3 P L O T                  */
  /*                                                     */
  /*******************************************************/

  /* CD_D3plot DEALLOC */
  static void
  CD_D3plot_dealloc(CD_D3plot* self)
  {

    if(self->d3plot != NULL){
      delete self->d3plot;
      self->d3plot = NULL;
    }
	
	#ifdef CD_DEBUG
	cout << "D3plot destructor" << endl;
	#endif

  }

  /* CD_D3plot NEW */
  static PyObject *
  CD_D3plot_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
  {

    CD_D3plot* self;
    self = (CD_D3plot *)type->tp_alloc(type, 0);

    // Init vars if any ...
    if (self != NULL){
      self->d3plot = NULL;
    }

    return (PyObject*) self;

  }


  /* CD_D3plot INIT */
  static int
  CD_D3plot_init(CD_D3plot *self, PyObject *args, PyObject *kwds)
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
        self->d3plot = new D3plot(string(filepath_c),(bool) useFemzip, variables);
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


  /* CD_D3plot FUNCTION get_timesteps */
  static PyObject *
  CD_D3plot_get_timesteps(CD_D3plot* self){

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

  /* CD_D3plot FUNCTION get_filepath */
  static PyObject *
  CD_D3plot_get_filepath(CD_D3plot* self){

    if (self->d3plot == NULL) {
        PyErr_SetString(PyExc_AttributeError, "Developer Error d3plot pointer NULL.");
        return NULL;
    }

    return Py_BuildValue("s",self->d3plot->get_filepath().c_str());

  }

  /* CD_D3plot FUNCTION read_states */
  static PyObject *
  CD_D3plot_read_states(CD_D3plot* self, PyObject* args){

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

  /* CD_D3plot FUNCTION get_nodeByID */
  static PyObject *
  CD_D3plot_get_nodeByID(CD_D3plot* self, PyObject* args){

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
      PyObject* ret = PyObject_CallObject((PyObject *) &CD_Node_Type, argList2);
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
        PyObject* ret = PyObject_CallObject((PyObject *) &CD_Node_Type, argList2);
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


  /* CD_D3plot FUNCTION get_elementByID */
  static PyObject *
  CD_D3plot_get_elementByID(CD_D3plot* self, PyObject* args){

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
      PyObject* ret = PyObject_CallObject((PyObject *) &CD_Element_Type, argList2);
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
        PyObject* ret = PyObject_CallObject((PyObject *) &CD_Element_Type, argList2);
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


  /* CD_D3plot FUNCTION get_partByID */
  static PyObject *
  CD_D3plot_get_partByID(CD_D3plot* self, PyObject* args){

    if (self->d3plot == NULL) {
      PyErr_SetString(PyExc_AttributeError, "Developer Error d3plot pointer NULL.");
      return NULL;
    }

    int partID;
    if (!PyArg_ParseTuple(args, "i", &partID))
      return NULL;

    PyObject *argList2 = Py_BuildValue("Oi",self , partID);
    PyObject* ret = PyObject_CallObject((PyObject *) &CD_Part_Type, argList2);
    Py_DECREF(argList2);

    return ret;

  }

  /* CD_D3plot FUNCTION get_parts */
  static PyObject *
  CD_D3plot_get_parts(CD_D3plot* self, PyObject* args){
  
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
      PyObject* ret = PyObject_CallObject((PyObject *) &CD_Part_Type, argList2);
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
  

  /*******************************************************/
  /*                                                     */
  /*                    C D _ N O D E                    */
  /*                                                     */
  /*******************************************************/


  /* CD_Node DEALLOC */
  static void
  CD_Node_dealloc(CD_Node* self)
  {

    Py_DECREF(self->d3plot_py);

  }


  /* CD_Node NEW */
  static PyObject *
  CD_Node_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
  {

    CD_Node* self;
    self = (CD_Node *)type->tp_alloc(type, 0);

    // Init vars if any ...
    if (self != NULL){
      self->node = NULL;
    }


    return (PyObject*) self;

  }


  /* CD_Node INIT */
  static int
  CD_Node_init(CD_Node *self, PyObject *args, PyObject *kwds)
  {

    PyObject* d3plot_obj_py;
    CD_D3plot* d3plot_py;
    int nodeID;
    static char *kwlist[] = {"d3plot","nodeID", NULL}; // TODO Deprecated!

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "Oi", kwlist, &d3plot_obj_py, &nodeID)){
        return -1;
    }

    if (! PyObject_TypeCheck(d3plot_obj_py, &CD_D3plot_Type)) {
      PyErr_SetString(PyExc_SyntaxError, "arg #1 not a d3plot in node constructor");
      return -1;
    }

    d3plot_py = (CD_D3plot*) d3plot_obj_py;

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

  /* CD_Node FUNCTION get_NodeID */
  static PyObject *
  CD_Node_get_NodeID(CD_Node* self){

    if(self->node == NULL){
      PyErr_SetString(PyExc_AttributeError,"Pointer to node is NULL.");
      return NULL;
    }

    int nodeID = self->node->get_nodeID();

    return Py_BuildValue("i",nodeID);

  }


  /* CD_Node FUNCTION get_coords */
  static PyObject *
  CD_Node_get_coords(CD_Node* self, PyObject *args, PyObject *kwds){

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

  /* CD_Node FUNCTION get_disp */
  static PyObject *
  CD_Node_get_disp(CD_Node* self){

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


  /* CD_Node FUNCTION get_vel */
  static PyObject *
  CD_Node_get_vel(CD_Node* self){

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


  /* CD_Node FUNCTION get_accel */
  static PyObject *
  CD_Node_get_accel(CD_Node* self){

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

  /* CD_Node FUNCTION get_elements */
  static PyObject *
  CD_Node_get_elements(CD_Node* self){

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
      PyObject* ret = PyObject_CallObject((PyObject *) &CD_Element_Type, argList2);
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


  /*******************************************************/
  /*                                                     */
  /*                 C D _ E L E M E N T                 */
  /*                                                     */
  /*******************************************************/

  /* CD_Element DEALLOC */
  static void
  CD_Element_dealloc(CD_Element* self)
  {

    Py_DECREF(self->d3plot_py);

  }

  /* CD_Element NEW */
  static PyObject *
  CD_Element_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
  {

    CD_Element* self;
    self = (CD_Element *)type->tp_alloc(type, 0);

    // Init vars if any ...
    if (self != NULL){
      self->element = NULL;
    }


    return (PyObject*) self;

  }


  /* CD_Element INIT */
  static int
  CD_Element_init(CD_Element *self, PyObject *args, PyObject *kwds)
  {

    PyObject* d3plot_obj_py;
    char* elementType_c;
    int elementID;
    static char *kwlist[] = {"d3plot","elementType","elementID", NULL}; // TODO Deprecated!

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "Osi", kwlist, &d3plot_obj_py, &elementType_c, &elementID)){
        return -1;
    }

    if (! PyObject_TypeCheck(d3plot_obj_py, &CD_D3plot_Type)) {
      PyErr_SetString(PyExc_TypeError, "arg #1 not a d3plot in element constructor");
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

    CD_D3plot* d3plot_py = (CD_D3plot*) d3plot_obj_py;

    if(d3plot_py->d3plot == NULL){
      PyErr_SetString(PyExc_AttributeError,"Pointer to d3plot-object is NULL.");
      return -1;
    }

    self->d3plot_py = d3plot_py;
    Py_INCREF(self->d3plot_py);
    self->element = d3plot_py->d3plot->get_db_elements()->get_elementByID(elementType,elementID);

    if(self->element == NULL){
      PyErr_SetString(PyExc_RuntimeError,string("Could not find any element with ID: "+to_string(elementID)+".").c_str());
      return -1;
    }

    return 0;
  }


  /* CD_Element FUNCTION get_id */
  static PyObject *
  CD_Element_get_elementID(CD_Element* self){

    if(self->element == NULL){
      PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
      return NULL;
    }

    int elementID = self->element->get_elementID();

    return Py_BuildValue("i",elementID);
  }


  /* CD_Element FUNCTION get_plastic_strain */
  static PyObject *
  CD_Element_get_plastic_strain(CD_Element* self){

    if(self->element == NULL){
      PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
      return NULL;
    }

    vector<float> plastic_strain = self->element->get_plastic_strain();

    int check = 0;
    PyObject* plastic_strain_list = PyList_New(plastic_strain.size());
    for(unsigned int ii=0; ii<plastic_strain.size(); ii++){
      check += PyList_SetItem(plastic_strain_list, ii,Py_BuildValue("f",plastic_strain[ii]));
    }

    if(check != 0){
      Py_DECREF(plastic_strain_list);
      PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of plastic-strain list.");
      return NULL;
    }

    return plastic_strain_list;

  }


  /* CD_Element FUNCTION get_energy */
  static PyObject *
  CD_Element_get_energy(CD_Element* self){

    if(self->element == NULL){
      PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
      return NULL;
    }

    vector<float> energy = self->element->get_energy();

    int check = 0;
    PyObject* energy_list = PyList_New(energy.size());
    for(unsigned int ii=0; ii<energy.size(); ii++){
      check += PyList_SetItem(energy_list, ii,Py_BuildValue("f",energy[ii]));
    }

    if(check != 0){
      Py_DECREF(energy_list);
      PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of energy list.");
      return NULL;
    }

    return energy_list;

  }


  /* CD_Element FUNCTION get_strain */
  static PyObject *
  CD_Element_get_strain(CD_Element* self){

    if(self->element == NULL){
      PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
      return NULL;
    }

    vector< vector<float> > strain = self->element->get_strain();

    int check0 = 0;
    int check1 = 0;
    PyObject* strain_time_list = PyList_New(strain.size());

    for(unsigned int ii=0; ii<strain.size(); ii++){

      PyObject* strain_list = PyList_New(strain[ii].size());

      for(unsigned int jj=0; jj<strain[ii].size(); jj++){
        check1 += PyList_SetItem(strain_list, jj, Py_BuildValue("f",strain[ii][jj]));
      }

      check0 += PyList_SetItem(strain_time_list, ii, strain_list);
    }

    if( (check0 != 0) | (check1 != 0) ){
      /*
      for (int ii = 0; ii < PyList_Size(disp_time_list); ii++){
        PyObject* disp_list = PyList_GetItem(disp_time_list, ii);
        Py_DECREF(disp_list);
      }
      */
      Py_DECREF(strain_time_list); // TODO: What about the lists in the list?
      PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of strain list.");
      return NULL;
    }

    return strain_time_list;

  }


  /* CD_Element FUNCTION get_stress */
  static PyObject *
  CD_Element_get_stress(CD_Element* self){

    if(self->element == NULL){
      PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
      return NULL;
    }

    vector< vector<float> > stress = self->element->get_stress();

    int check0 = 0;
    int check1 = 0;
    PyObject* stress_time_list = PyList_New(stress.size());

    for(unsigned int ii=0; ii<stress.size(); ii++){

      PyObject* stress_list = PyList_New(stress[ii].size());

      for(unsigned int jj=0; jj<stress[ii].size(); jj++){
        check1 += PyList_SetItem(stress_list, jj, Py_BuildValue("f",stress[ii][jj]));
      }

      check0 += PyList_SetItem(stress_time_list, ii, stress_list);
    }

    if( (check0 != 0) | (check1 != 0) ){
      /*
      for (int ii = 0; ii < PyList_Size(disp_time_list); ii++){
        PyObject* disp_list = PyList_GetItem(disp_time_list, ii);
        Py_DECREF(disp_list);
      }
      */
      Py_DECREF(stress_time_list); // TODO: What about the lists in the list?
      PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of strain list.");
      return NULL;
    }

    return stress_time_list;

  }

  /* CD_Element FUNCTION get_nodes */
  static PyObject *
  CD_Element_get_nodes(CD_Element* self){

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

      PyObject *argList2 = Py_BuildValue("Oi",self->d3plot_py ,node->get_nodeID());
      PyObject* ret = PyObject_CallObject((PyObject *) &CD_Node_Type, argList2);
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

  /* CD_Element FUNCTION get_coords */
  static PyObject *
  CD_Element_get_coords(CD_Element* self, PyObject *args, PyObject *kwds){
  
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

    vector<float> coords;
    try{
      coords = self->element->get_coords(iTimestep);
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
  
  /* CD_Element FUNCTION get_history */
  static PyObject *
  CD_Element_get_history(CD_Element* self){
     
    if(self->element == NULL){
      PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
      return NULL;
    }
    
    vector< vector<float> > history_vars = self->element->get_history_vars();
    
    int check0 = 0;
    int check1 = 0;
    PyObject* history_vars_list0 = PyList_New(history_vars.size());
    for(unsigned int ii=0; ii<history_vars.size(); ii++){

      PyObject* history_vars_list1 = PyList_New(history_vars[ii].size());

      for(unsigned int jj=0; jj<history_vars[ii].size(); jj++){
        check1 += PyList_SetItem(history_vars_list1, jj, Py_BuildValue("f",history_vars[ii][jj]));
      }
      check0 += PyList_SetItem(history_vars_list0, ii, history_vars_list1);
    }

    if( (check0 != 0) | (check1 != 0) ){
      Py_DECREF(history_vars_list0); // TODO: What about the lists in the list?
      PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of strain list.");
      return NULL;
    }

    return history_vars_list0;
     
  }
  
  /* CD_Element FUNCTION get_estimated_size */
  static PyObject *
  CD_Element_get_estimated_size(CD_Element* self){
  
    if(self->element == NULL){
      PyErr_SetString(PyExc_AttributeError,"Pointer to element is NULL.");
      return NULL;
    }
        
    return Py_BuildValue("f",self->element->get_estimated_element_size());
  
  }
   

  /* CD_Element FUNCTION get_type */
  static PyObject *
  CD_Element_get_type(CD_Element* self){
  
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
  
  
  /*******************************************************/
  /*                                                     */
  /*                     Q D _ P A R T                   */
  /*                                                     */
  /*******************************************************/

  /* CD_Part DEALLOC */
  static void
  CD_Part_dealloc(CD_Part* self){
    Py_DECREF(self->d3plot_py);
  }


  /* CD_Part NEW */
  static PyObject *
  CD_Part_new(PyTypeObject *type, PyObject *args, PyObject *kwds){

    CD_Part* self;
    self = (CD_Part *)type->tp_alloc(type, 0);

    // Init vars if any ...
    if (self != NULL){
      self->part = NULL;
    }


    return (PyObject*) self;

  }

  /* CD_Part INIT */
  static int
  CD_Part_init(CD_Part *self, PyObject *args, PyObject *kwds){

    PyObject* d3plot_obj_py;
    int partID;
    static char *kwlist[] = {"d3plot","partID", NULL}; // TODO Deprecated!

    
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "Oi", kwlist, &d3plot_obj_py, &partID)){
        return -1;
    }

    if (! PyObject_TypeCheck(d3plot_obj_py, &CD_D3plot_Type)) {
      PyErr_SetString(PyExc_TypeError, "arg #1 not a d3plot in part constructor");
      return -1;
    }
    CD_D3plot* d3plot_py = (CD_D3plot*) d3plot_obj_py;

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
  CD_Part_get_id(CD_Part *self){

    if(self->part == NULL){
      PyErr_SetString(PyExc_AttributeError,"Pointer to part is NULL.");
      return NULL;
    }

    return Py_BuildValue("i",self->part->get_partID());

  }


  /* CD_Part FUNCTION get_name */
  static PyObject*
  CD_Part_get_name(CD_Part *self){

    if(self->part == NULL){
      PyErr_SetString(PyExc_AttributeError,"Pointer to part is NULL.");
      return NULL;
    }

    string partName = self->part->get_name();

    return Py_BuildValue("s",partName.c_str());

  }


  /* CD_Part FUNCTION get_nodes */
  static PyObject*
  CD_Part_get_nodes(CD_Part *self){

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
      PyObject* ret = PyObject_CallObject((PyObject *) &CD_Node_Type, argList2);
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


  /* CD_Part FUNCTION get_elements */
  static PyObject*
  CD_Part_get_elements(CD_Part *self){

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
      PyObject* ret = PyObject_CallObject((PyObject *) &CD_Element_Type, argList2);
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


  /*******************************************************/
  /*                                                     */
  /*                   Q D _ M O D U L E                 */
  /*                                                     */
  /*******************************************************/

  static PyObject *
  test_codie(PyObject *self, PyObject *args)
  {
	return Py_None;
  }
  
  /* MODULE codie function table */
  static PyMethodDef CodieMethods[] = {
    {"test_codie",  test_codie, METH_VARARGS,"Execute a shell command."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
  };
  
  /* MODULE codie */
  /* PYTHON 3 STUFF ?!?!?
  static PyModuleDef codie_module = {
    PyModuleDef_HEAD_INIT,
    "codie",
    "Codie Python cae module.",
    -1,
	CodieMethods
    //NULL, NULL, NULL, NULL, NULL
  };
  */
  
  /* MODULE INIT codie */
  /* PY3
  PyMODINIT_FUNC
  PyInit_codie(void)
  */
  void
  //initcodie(void)
  initdyna(void)
  {
    PyObject* m;

    // Constructor
    if (PyType_Ready(&CD_D3plot_Type) < 0)
      // PY3 return NULL;
      return;

    if (PyType_Ready(&CD_Node_Type) < 0)
      // PY3 return NULL;
      return;

    if (PyType_Ready(&CD_Element_Type) < 0)
      // PY3 return NULL;
      return;

    if (PyType_Ready(&CD_Part_Type) < 0)
      // PY3 return NULL;
      return;

    // Init Module
    // Python 2.7
    m = Py_InitModule3("dyna", CodieMethods,
                       "Dyna routines for qd cae.");
    // PY3 m = PyModule_Create(&codie_module);
    /*
    if (import_cd_cae() < 0)
      return NULL;
    */
    if (m == NULL)
      // PY3 return NULL;
      return;

    Py_INCREF(&CD_D3plot_Type);
    PyModule_AddObject(m, "D3plot", (PyObject *)&CD_D3plot_Type);

    Py_INCREF(&CD_Node_Type);
    PyModule_AddObject(m, "Node", (PyObject *)&CD_Node_Type);

    Py_INCREF(&CD_Element_Type);
    PyModule_AddObject(m, "Element", (PyObject *)&CD_Element_Type);

    Py_INCREF(&CD_Element_Type);
    PyModule_AddObject(m, "Part", (PyObject *)&CD_Part_Type);

    // PY3 return m;
  }

}
