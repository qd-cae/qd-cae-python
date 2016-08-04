
#ifndef ABSTRACTBYTEBUFFER
#define ABSTRACTBYTEBUFFER

#include <string.h>

using namespace std;

class AbstractBuffer{

  public:
  // Standard
  //AbstractBuffer(){};
  virtual ~AbstractBuffer(){};
  // Geometry
  virtual void read_geometryBuffer()=0;
  virtual void free_geometryBuffer()=0;
  // Parts
  virtual void read_partBuffer()=0;
  virtual void free_partBuffer()=0;
  // States
  virtual void init_nextState()=0;
  virtual void read_nextState()=0;
  virtual bool has_nextState()=0;
  virtual void rewind_nextState()=0;
  virtual void end_nextState()=0;
  // Close 
  virtual void finish_reading()=0;

  // Vars
  virtual int read_int(int)=0;
  virtual float read_float(int)=0;
  virtual string read_str(int,int)=0;

};


#endif
