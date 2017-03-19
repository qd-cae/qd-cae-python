
#ifndef D3PLOTBUFFER_HPP
#define D3PLOTBUFFER_HPP

#include <string>
#include <future>
#include "AbstractBuffer.hpp"

using namespace std;

class D3plotBuffer : public AbstractBuffer {

  private:
  unsigned int iStateFile;
  vector<char> current_buffer;
  vector< future< vector<char> > > state_buffers; // preloaded states (REVERSED!!!)
  int wordSize;
  long bufferSize;
  vector<string> d3plots;
  static vector<char> get_bufferFromFile(string); // helper function

  public:
  D3plotBuffer(string _d3plot_path, int _wordSize);
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
