
#ifndef D3PLOTBUFFER
#define D3PLOTBUFFER

#include <string>
#include "AbstractBuffer.h"

using namespace std;

class D3plotBuffer : public AbstractBuffer {

  private:
  unsigned int iStateFile;
  char* current_buffer;
  int wordSize;
  long bufferSize;
  vector<string> d3plots;
  void get_bufferFromFile(string);

  public:
  D3plotBuffer(string,int);
  ~D3plotBuffer();
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
