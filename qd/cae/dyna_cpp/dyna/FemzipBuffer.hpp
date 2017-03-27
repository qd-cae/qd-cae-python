
#ifndef FEMZIPBUFFER_HPP
#define FEMZIPBUFFER_HPP

#include <string>
#include <future>
#include "dyna_cpp/dyna/AbstractBuffer.hpp"

class FemzipBuffer : public AbstractBuffer {

  /* PRIVATE */
  private:
  std::string filepath;

  std::future< std::vector<char> > next_state_buffer;

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

  void check_ier(std::string);
  void init_vars();

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

#endif
