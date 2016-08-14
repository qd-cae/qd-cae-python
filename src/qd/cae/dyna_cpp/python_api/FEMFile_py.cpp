

/* DEALLOC */
static void
QD_FEMFile_dealloc(QD_FEMFile* self)
{
   if(self->femfile != NULL){
      delete self->femfile;
      self->femfile = NULL;
   }
}

/* NEW */
static PyObject *
QD_FEMFile_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{

  QD_FEMFile* self;
  self = (QD_FEMFile *)type->tp_alloc(type, 0);

  // Init vars if any ...
  if (self != NULL){
    self->femfile = NULL;
  }

  return (PyObject*) self;

}

/* INIT */
static int
QD_FEMFile_init(QD_FEMFile *self, PyObject *args, PyObject *kwds)
{

  return 0;
}
