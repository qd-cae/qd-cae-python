

/* QD_KeyFile DEALLOC */
static void
QD_KeyFile_dealloc(QD_KeyFile* self)
{

  if(self->keyFile != NULL){
    delete self->keyFile;
    self->keyFile = NULL;
    self->femfile.instance = NULL; // = self->keyFile
  }

 #ifdef QD_DEBUG
 cout << "KeyFile destructor" << endl;
 #endif

}

/* QD_KeyFile NEW */
static PyObject *
QD_KeyFile_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

  QD_KeyFile* self;
  self = (QD_KeyFile *)type->tp_alloc(type, 0);

  // Init vars if any ...
  if (self != NULL){
    self->keyFile = NULL;
    self->femfile.instance = NULL;
  }

  return (PyObject*) self;
}


/* QD_KeyFile INIT */
static int
QD_KeyFile_init(QD_KeyFile *self, PyObject *args, PyObject *kwds)
{

   char* filepath_c = NULL;
   static char *kwlist[] = {"filepath",NULL};

   if (! PyArg_ParseTupleAndKeywords(args, kwds, "|s", kwlist, &filepath_c)){
      return -1;
   }

   try{
      if(filepath_c){
         self->keyFile = new KeyFile(string(filepath_c));
      } else {
         self->keyFile = new KeyFile();
      }
      self->femfile.instance = self->keyFile;
   } catch (string e){
      PyErr_SetString(PyExc_RuntimeError, e.c_str());
      return -1;
   }

   return 0;

}
