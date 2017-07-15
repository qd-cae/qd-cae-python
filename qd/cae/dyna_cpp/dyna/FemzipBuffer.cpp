
#include <dyna_cpp/dyna/FemzipBuffer.hpp>
#include <dyna_cpp/utility/FileUtility.hpp>

#include <bitset>
#include <iostream>
#include <sstream>
#include <stdexcept>

extern "C" {
#include "dyna_cpp/dyna/femzip.h"
#include <stdio.h>
}

namespace qd {

/*
 * Constructor
 */
FemzipBuffer::FemzipBuffer(std::string _filepath)
  : AbstractBuffer(4)
  , filetype(1)
  , ier(0)
  , pos(0)
  , size_geo(0)
  , size_state(0)
  , size_disp(0)
  , size_activity(0)
  , size_post(0)
  , size_titles(0)
  , iTimeStep(1) // ... why :(
  , size_times(0)
  , adjust(5)
{

  // Init vars
  this->filepath = _filepath;
  if (!FileUtility::check_ExistanceAndAccess(this->filepath)) {
    throw(std::invalid_argument("File \"" + this->filepath +
                                "\" does not exist or is locked."));
  }

  // version check
  float unzipversion = 0.;
  float fileunzipversion = 0.;

  femunziplib_version(&unzipversion);
  femunziplib_version_file((char*)this->filepath.c_str(),
                           &this->filetype,
                           &fileunzipversion,
                           &this->ier);
  this->check_ier("Femzip Error during femunziplib_version_file.");

  if (unzipversion < fileunzipversion) {
    throw(std::invalid_argument("Femzip version older than file version."));
  }

  /* SIZES */
  get_Size((char*)this->filepath.c_str(),
           this->filetype,
           this->adjust,
           &this->size_geo,
           &this->size_state,
           &this->size_disp,
           &this->size_activity,
           &this->size_post,
           &this->ier);
  this->check_ier("Femzip Error during reading of sizes.");
}

/*
 * Destructor
 */
FemzipBuffer::~FemzipBuffer()
{
}

/*
 * Read the geometry buffer.
 */
void
FemzipBuffer::read_geometryBuffer()
{

  std::string nonsense_s = "nonsense";
  char* nonsense = (char*)nonsense_s.c_str();
  char* argv[] = { nonsense, (char*)this->filepath.c_str() };
  int32_t p1[1000];
  int32_t p2[1000];
  int32_t l1 = 0;
  int32_t l2 = 0;
  wrapinput(2, argv, p1, p2, &l1, &l2);

  this->current_buffer.reserve(sizeof(int32_t) * this->size_geo);
  geometry_read(p1,
                p2,
                &l1,
                &l2,
                &this->ier,
                &this->pos,
                (int32_t*)&this->current_buffer[0],
                &this->size_geo);
  this->check_ier("Femzip Error while reading geometry.");
}

/*
 * Free the geometry buffer.
 */
void
FemzipBuffer::free_geometryBuffer()
{
}

/*
 * Read the part buffer.
 */
void
FemzipBuffer::read_partBuffer()
{

  this->pos = 0;
  part_titles_read(&this->ier,
                   &this->pos,
                   (int32_t*)&this->current_buffer[0],
                   &this->size_titles);
  this->current_buffer.reserve(sizeof(int32_t) * this->size_geo);
  this->pos = 0;
  part_titles_read(&this->ier,
                   &this->pos,
                   (int32_t*)&this->current_buffer[0],
                   &this->size_titles);
  check_ier("Femzip Error during part_titles_read.");
}

/*
 * Free the part buffer.
 */
void
FemzipBuffer::free_partBuffer()
{
}

/*
 * Init the state reading.
 */
void
FemzipBuffer::init_nextState()
{

  this->iTimeStep = 1;
  int32_t retry = 0;
  int32_t size_times = 2000;
retry:
  this->timese = new float[size_times];
  this->pos = 0;
  ctimes_read(
    &this->ier, &this->pos, &this->nTimeStep, this->timese, &size_times);
  if (this->ier == 9) {
    if (retry < 1) {
      retry++;
      goto retry;
    }
    if (this->timese != NULL) {
      delete[] this->timese;
      this->timese = NULL;
    }
    throw(std::invalid_argument(
      "Femzip Buffer Error: size for the states buffer is to small."));
  }

  if (this->timese != NULL) {
    delete[] this->timese;
    this->timese = NULL;
  }
  check_ier("Femzip Error during ctimes_read.");

  this->timese = NULL;

  // preload timestep
  this->next_state_buffer = std::async(
    [](int32_t _iTimestep, int32_t _size_state) {
      int32_t _ier = 0;
      int32_t _pos = 0;
      std::vector<char> state_buffer(sizeof(int32_t) * _size_state);
      states_read(
        &_ier, &_pos, &_iTimestep, (int32_t*)&state_buffer[0], &_size_state);
      if (_ier != 0) {
        if (state_buffer.size() != 0) {
          state_buffer = std::vector<char>();
        }
      }

      return std::move(state_buffer);
    },
    this->iTimeStep,
    this->size_state);
}

/*
 * Init the state reading.
 */
void
FemzipBuffer::read_nextState()
{

#ifdef QD_DEBUG
  cout << "Loading state: " << this->iTimeStep << "/" << this->nTimeStep
       << endl;
#endif

  this->current_buffer = this->next_state_buffer.get();
  if (this->current_buffer.size() == 0) {
    throw(std::invalid_argument("FEMZIP Error during state reading."));
  }

  this->iTimeStep++;

  // preload timestep
  this->next_state_buffer = std::async(
    [](int32_t _iTimestep, int32_t _size_state) {
      int32_t _ier = 0;
      int32_t _pos = 0;
      std::vector<char> state_buffer(sizeof(int32_t) * _size_state);
      states_read(
        &_ier, &_pos, &_iTimestep, (int32_t*)&state_buffer[0], &_size_state);
      if (_ier != 0) {
        if (state_buffer.size() != 0) {
          state_buffer = std::vector<char>();
        }
      }

      return std::move(state_buffer);
    },
    this->iTimeStep,
    this->size_state);
}

/*
 * Is there another state?
 */
bool
FemzipBuffer::has_nextState()
{
  return this->iTimeStep <= this->nTimeStep;
}

/*
 * Is there another state?
 */
void
FemzipBuffer::rewind_nextState()
{

  this->iTimeStep = 1;

  // Size
  this->pos = 0;
  get_Size((char*)this->filepath.c_str(),
           this->filetype,
           this->adjust,
           &this->size_geo,
           &this->size_state,
           &this->size_disp,
           &this->size_activity,
           &this->size_post,
           &this->ier);
  this->check_ier("Femzip Error during reading of sizes.");
  // geom must be read again ...
  this->read_geometryBuffer();
  this->free_geometryBuffer();
  // parts not ... at least
  // ...
  // states
  this->init_nextState();
}

/*
 * End the state reading.
 */
void
FemzipBuffer::end_nextState()
{

  states_close(&this->ier,
               &this->pos,
               (int32_t*)&this->current_buffer[0],
               &this->size_geo);
  current_buffer.clear();

  close_read(&this->ier);
  this->check_ier("Femzip Error during state of file.");
}

/*
 * Close the file.
 */
void
FemzipBuffer::finish_reading()
{

  close_read(&this->ier);
  this->check_ier("Femzip Error during closing of file.");
}

/*
 * Check for error
 */
void
FemzipBuffer::check_ier(std::string message)
{
  if (this->ier != 0) {
    throw(std::invalid_argument(message));
  }
}

} // anemspace qd
