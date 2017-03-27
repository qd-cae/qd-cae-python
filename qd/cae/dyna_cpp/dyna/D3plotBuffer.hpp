
#ifndef D3PLOTBUFFER_HPP
#define D3PLOTBUFFER_HPP

#include <string>
#include <vector>
#include <future>
#include "dyna_cpp/dyna/AbstractBuffer.hpp"

class D3plotBuffer : public AbstractBuffer {

  private:
  unsigned int iStateFile;
  std::vector< std::future< std::vector<char> > > state_buffers; // preloaded states (REVERSED!!!)
  long bufferSize;
  std::vector<std::string> d3plots;
  static std::vector<char> get_bufferFromFile(std::string); // helper function

  public:
  D3plotBuffer(std::string _d3plot_path, int _wordSize);
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

};


#endif
