

/* DEALLOC */
static void QD_Element_dealloc(QD_Element *self) {

  Py_DECREF(self->femFile_py);
}

/* NEW */
static PyObject *QD_Element_new(PyTypeObject *type, PyObject *args,
                                PyObject *kwds) {

  QD_Element *self;
  self = (QD_Element *)type->tp_alloc(type, 0);

  // Init vars if any ...
  if (self != nullptr) {
    self->element = nullptr;
  }

  return (PyObject *)self;
}

/* INIT */
static int QD_Element_init(QD_Element *self, PyObject *args, PyObject *kwds) {

  PyObject *femfile_obj_py;
  char *elementType_c;
  int elementID;
  static char *kwlist[] = {
      const_cast<char *>("femfile"), const_cast<char *>("elementType"),
      const_cast<char *>("elementID"), nullptr}; // TODO Deprecated!

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "Osi", kwlist, &femfile_obj_py,
                                   &elementType_c, &elementID)) {
    return -1;
  }

  if (!PyObject_TypeCheck(femfile_obj_py, &QD_FEMFile_Type)) {
    PyErr_SetString(PyExc_TypeError,
                    "arg #1 not a D3plot or KeyFile in element constructor");
    return -1;
  }

  ElementType elementType;
  string elementType_s(elementType_c);
  if(elementType_s.find("beam") != string::npos) {
    elementType = BEAM;
  } else if (elementType_s.find("shell") != string::npos) {
    elementType = SHELL;
  } else if (elementType_s.find("solid") != string::npos) {
    elementType = SOLID;
  } else {
    PyErr_SetString(PyExc_SyntaxError,
                    "Unknown element-type. Try: beam, shell, solid.");
    return -1;
  }

  QD_FEMFile *femFile_py = (QD_FEMFile *)femfile_obj_py;

  if (femFile_py->instance == nullptr) {
    PyErr_SetString(PyExc_AttributeError,
                    "Pointer to C++ File-Object is nullptr.");
    return -1;
  }

  self->femFile_py = femFile_py;
  Py_INCREF(self->femFile_py);
  self->element = femFile_py->instance->get_db_elements()->get_elementByID(
      elementType, elementID);

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_RuntimeError,
                    string("Could not find any element with ID: " +
                           to_string(elementID) + ".")
                        .c_str());
    return -1;
  }

  return 0;
}

/* FUNCTION richcompare */
static PyObject *QD_Element_richcompare(QD_Element *self, PyObject *other,
                                        int op) {

  PyObject *result = nullptr;

  if (!PyObject_TypeCheck(other, &QD_Element_Type)) {
    PyErr_SetString(PyExc_ValueError,
                    "Comparison of elements work only with other elements.");
    return nullptr;
  }

  QD_Element *other_elem = (QD_Element *)other;

  switch (op) {
  case Py_LT:
    if (self->element->get_elementID() < other_elem->element->get_elementID()) {
      result = Py_True;
    } else {
      result = Py_False;
    }
    break;
  case Py_LE:
    if (self->element->get_elementID() <=
        other_elem->element->get_elementID()) {
      result = Py_True;
    } else {
      result = Py_False;
    }
    break;
  case Py_EQ:
    if (self->element->get_elementID() ==
            other_elem->element->get_elementID() &&
        self->element->get_elementType() ==
            other_elem->element->get_elementType()) {
      result = Py_True;
    } else {
      result = Py_False;
    }
    break;
  case Py_NE:
    if (self->element->get_elementID() !=
            other_elem->element->get_elementID() ||
        self->element->get_elementType() !=
            other_elem->element->get_elementType()) {
      result = Py_True;
    } else {
      result = Py_False;
    }
    break;
  case Py_GT:
    if (self->element->get_elementID() > other_elem->element->get_elementID()) {
      result = Py_True;
    } else {
      result = Py_False;
    }
    break;
  case Py_GE:
    if (self->element->get_elementID() >=
        other_elem->element->get_elementID()) {
      result = Py_True;
    } else {
      result = Py_False;
    }
    break;
  }

  Py_XINCREF(result);
  return result;
}

/* FUNCTION get_id */
static PyObject *QD_Element_get_elementID(QD_Element *self) {

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "Pointer to element is nullptr.");
    return nullptr;
  }

  int elementID = self->element->get_elementID();

  return Py_BuildValue("i", elementID);
}

/* FUNCTION get_plastic_strain */
static PyObject *QD_Element_get_plastic_strain(QD_Element *self) {

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "Pointer to element is nullptr.");
    return nullptr;
  }

  return (PyObject *)vector_to_nparray(self->element->get_plastic_strain());
}

/* FUNCTION get_energy */
static PyObject *QD_Element_get_energy(QD_Element *self) {

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "Pointer to element is nullptr.");
    return nullptr;
  }

  return (PyObject *)vector_to_nparray(self->element->get_energy());
}

/* FUNCTION get_strain */
static PyObject *QD_Element_get_strain(QD_Element *self) {

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "Pointer to element is nullptr.");
    return nullptr;
  }

  return (PyObject *)vector_to_nparray(self->element->get_strain());
}

/* FUNCTION get_stress */
static PyObject *QD_Element_get_stress(QD_Element *self) {

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "Pointer to element is nullptr.");
    return nullptr;
  }

  return (PyObject *)vector_to_nparray(self->element->get_stress());
}

/* FUNCTION get_stress_mises */
static PyObject *QD_Element_get_stress_mises(QD_Element *self) {

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "Pointer to element is nullptr.");
    return nullptr;
  }

  return (PyObject *)vector_to_nparray(self->element->get_stress_mises());
}

/* FUNCTION get_nodes */
static PyObject *QD_Element_get_nodes(QD_Element *self) {

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "Pointer to element is nullptr.");
    return nullptr;
  }

  vector<Node *> nodes = self->element->get_nodes();

  vector<Node *>::iterator it;
  int check = 0;
  PyObject *node_list = PyList_New(nodes.size());

  unsigned int ii = 0;
  Node *node = nullptr;
  for (vector<Node *>::iterator it = nodes.begin(); it != nodes.end(); ++it) {
    node = *it;

    PyObject *argList2 =
        Py_BuildValue("Oi", self->femFile_py, node->get_nodeID());
    PyObject *ret = PyObject_CallObject((PyObject *)&QD_Node_Type, argList2);
    Py_DECREF(argList2);

    check += PyList_SetItem(node_list, ii, ret);

    ++ii;
  }

  if (check != 0) {
    Py_DECREF(node_list);
    PyErr_SetString(PyExc_RuntimeError,
                    "Developer Error during assembly of node instance list.");
    return nullptr;
  }

  return node_list;
}

/* FUNCTION get_coords */
static PyObject *QD_Element_get_coords(QD_Element *self, PyObject *args,
                                       PyObject *kwds) {

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "Pointer to element is nullptr.");
    return nullptr;
  }

  int iTimestep = 0;
  static char *kwlist[] = {const_cast<char *>("iTimestep"),
                           nullptr}; // TODO Deprecated!

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &iTimestep)) {
    return nullptr;
  }

  try {
    return (PyObject *)vector_to_nparray(self->element->get_coords(iTimestep));
  } catch (const string &e) {
    PyErr_SetString(PyExc_RuntimeError, e.c_str());
    return nullptr;
  }
}

/* FUNCTION get_history */
static PyObject *QD_Element_get_history(QD_Element *self) {

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "Pointer to element is nullptr.");
    return nullptr;
  }

  return (PyObject *)vector_to_nparray(self->element->get_history_vars());
}

/* FUNCTION get_estimated_size */
static PyObject *QD_Element_get_estimated_size(QD_Element *self) {

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "Pointer to element is nullptr.");
    return nullptr;
  }

  try {
    return Py_BuildValue("f", self->element->get_estimated_element_size());
  } catch (const string &err_message) {
    PyErr_SetString(PyExc_RuntimeError, err_message.c_str());
    return nullptr;
  } catch (const char *err_message_c) {
    PyErr_SetString(PyExc_RuntimeError, err_message_c);
    return nullptr;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "some unknown error occurred.");
    return nullptr;
  }
}

/* FUNCTION get_type */
static PyObject *QD_Element_get_type(QD_Element *self) {

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "Pointer to element is nullptr.");
    return nullptr;
  }

  ElementType type = self->element->get_elementType();
  if (type == SHELL) {
    return Py_BuildValue("s", "shell");
  } else if (type == SOLID) {
    return Py_BuildValue("s", "solid");
  } else if (type == BEAM) {
    return Py_BuildValue("s", "beam");
  } else {
    PyErr_SetString(PyExc_AttributeError, "Unknown element type detected.");
    return nullptr;
  }
}

/* FUNCTION get_is_rigid */
static PyObject *QD_Element_get_is_rigid(QD_Element *self) {

  if (self->element == nullptr) {
    PyErr_SetString(PyExc_AttributeError, "Pointer to element is nullptr.");
    return nullptr;
  }

  if (self->element->get_is_rigid()) {
    Py_INCREF(Py_True);
    return Py_True;
  } else {
    Py_INCREF(Py_False);
    return Py_False;
  }
}
