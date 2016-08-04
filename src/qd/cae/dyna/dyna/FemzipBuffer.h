
#ifndef FEMZIPBUFFER
#define FEMZIPBUFFER

#include <string>
#include "AbstractBuffer.h"

using namespace std;

class FemzipBuffer : public AbstractBuffer {

  /* PRIVATE */
  private:
  string filepath;
  int wordSize; // byte
  
  char* current_buffer;

  // general
  int filetype;// = 1;
  int ier;
  int pos;
  // Sizing
  int size_geo;
  int size_state;
  int size_disp;
  int size_activity;
  int size_post;
  int size_titles;
  // States
  int iTimeStep; // ... why
  int nTimeStep;
  int size_times;
  float* timese;
  // config
  int adjust;

  void check_ier(string);
  void init_vars();

  /* PUBLIC */
  public:
  FemzipBuffer(string);
  ~FemzipBuffer();
  void read_geometryBuffer();
  void free_geometryBuffer();
  // Parts
  void read_partBuffer();
  void free_partBuffer();
  // States
  void init_nextState();
  void read_nextState();
  bool has_nextState();
  void rewind_nextState();
  void end_nextState();
  // Close
  void finish_reading();

  // var reading
  int read_int(int);
  float read_float(int);
  string read_str(int,int);

};

#endif
