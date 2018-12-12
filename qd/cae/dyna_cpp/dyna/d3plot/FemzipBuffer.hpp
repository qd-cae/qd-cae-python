
#ifndef FEMZIPBUFFER_HPP
#define FEMZIPBUFFER_HPP

#include <deque>
#include <future>
#include <string>

#include <dyna_cpp/dyna/d3plot/AbstractBuffer.hpp>
// #include <dyna_cpp/parallel/WorkQueue.hpp>

namespace qd {

class FemzipBuffer : public AbstractBuffer
{
private:
  std::string filepath;
  std::future<std::vector<char>> _next_buffer;
  // std::deque<std::future<std::vector<char>>> _state_buffers;

  // WorkQueue _work_queue;

  // general
  int32_t filetype; // = 1;
  int32_t ier;
  int32_t pos;
  // Sizing
  int32_t size_geo;
  int32_t size_state;
  int32_t size_disp;
  int32_t size_activity;
  int32_t size_post;
  int32_t size_titles;
  // States
  int32_t iTimeStep; // ... why
  int32_t nTimeStep;
  int32_t size_times;
  float* timese;
  // config
  int32_t adjust;

  void check_ier(const std::string&);

  // helper function
  static std::vector<char> _load_next_timestep(int32_t _iTimestep,
                                               int32_t _size_state);

  /* PUBLIC */
public:
  FemzipBuffer(std::string);
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

  // static functions
  static bool is_femzipped(const std::string& filepath);
};

} // namespace qd

#endif
