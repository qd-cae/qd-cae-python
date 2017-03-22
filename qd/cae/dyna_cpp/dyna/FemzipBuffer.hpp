
#ifndef FEMZIPBUFFER_HPP
#define FEMZIPBUFFER_HPP

#include <string>
#include <future>
#include "AbstractBuffer.hpp"

class FemzipBuffer : public AbstractBuffer {

  /* PRIVATE */
  private:
  std::string filepath;
  int wordSize; // byte

  std::future<char*> next_state_buffer;
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

  // var reading
  int read_int(int);
  float read_float(int);
  std::string read_str(int,int);

};

#endif
