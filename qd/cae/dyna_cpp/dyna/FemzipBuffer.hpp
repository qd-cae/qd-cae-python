
#ifndef FEMZIPBUFFER_HPP
#define FEMZIPBUFFER_HPP

#include "dyna_cpp/dyna/AbstractBuffer.hpp"

#include <future>
#include <string>

namespace qd {

class FemzipBuffer : public AbstractBuffer
{

  /* PRIVATE */
private:
  std::string filepath;

  std::future<std::vector<char>> next_state_buffer;

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

  void check_ier(std::string);

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
};

} // namespace qd

#endif
