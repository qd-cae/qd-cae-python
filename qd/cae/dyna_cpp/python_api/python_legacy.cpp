

/*
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
//    for (int ii = 0; ii < PyList_Size(disp_time_list); ii++){
//      PyObject* disp_list = PyList_GetItem(disp_time_list, ii);
//      Py_DECREF(disp_list);
//    }
    Py_DECREF(disp_time_list); // TODO: What about the lists in the list?
    PyErr_SetString(PyExc_RuntimeError, "Developer Error during assembly of coords list.");
    return NULL;
  }

*/
