
#ifndef D3PLOTBUFFER_HPP
#define D3PLOTBUFFER_HPP

// includes
#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include <dyna_cpp/dyna/d3plot/AbstractBuffer.hpp>
// #include <dyna_cpp/parallel/WorkQueue.hpp>

namespace qd {

class D3plotBuffer : public AbstractBuffer
{

private:
  std::future<std::vector<char>> _next_buffer;

  // WorkQueue _work_queue;
  // std::deque<std::future<std::vector<char>>> _file_buffer_q;

  size_t iStateFile;
  size_t iActiveFile;
  std::vector<std::string> _d3plots;

  static std::vector<char> get_bufferFromFile(std::string); // helper function

public:
  explicit D3plotBuffer(std::string _d3plot_path, int32_t word_size);
  virtual ~D3plotBuffer();
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

} // namespace qd

#endif
