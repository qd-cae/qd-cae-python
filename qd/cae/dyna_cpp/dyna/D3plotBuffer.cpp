

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "dyna_cpp/dyna/D3plotBuffer.hpp"
#include "dyna_cpp/utility/FileUtility.hpp"

namespace qd {

/*
 * Constructor
 */
D3plotBuffer::D3plotBuffer(std::string _d3plot_path, int32_t _wordSize)
  : AbstractBuffer(_wordSize)
  , iStateFile(0)
  , bufferSize(0)
  , wordSize(_wordSize)
{

  // Check File
  if (!check_ExistanceAndAccess(_d3plot_path)) {
    throw(std::invalid_argument("File \"" + _d3plot_path +
                                "\" does not exist or is locked."));
  }

  this->d3plots = findDynaResultFiles(_d3plot_path);
#ifdef QD_DEBUG
  std::cout << "Found result files:" << endl;
  for (size_t ii = 0; ii < this->d3plots.size(); ++ii) {
    std::cout << this->d3plots[ii] << endl;
  }
  std::cout << "End of file list." << endl;
#endif
  // this->d3plots = globVector(_d3plot_path+"*");

  if (this->d3plots.size() < 1)
    throw(std::invalid_argument(
      "No D3plot result file could be found with the given path:" +
      _d3plot_path));
}

/*
 * Destructor
 */
D3plotBuffer::~D3plotBuffer()
{
  // clean up futures
  while (state_buffers.size() != 0) {
    state_buffers.back().get();
    state_buffers.pop_back();
  }
}

/*
 * Get a char* byte buffer from the given file.
 *
 */
std::vector<char>
D3plotBuffer::get_bufferFromFile(std::string filepath)
{

  std::vector<char> state_buffer;

  // Read data into buffer
  std::ifstream fStream;
  fStream.open(filepath.c_str(), std::ios::binary | std::ios::in);
  fStream.seekg(0, std::ios::end);
  std::streamoff _bufferSize = fStream.tellg();
  fStream.seekg(0, std::ios::beg);
  // std::cout << "Filesize: " << *bufferSize << endl; // DEBUG
  state_buffer.reserve(_bufferSize);
  fStream.read(&state_buffer[0], _bufferSize);
  fStream.close();

  return state_buffer;
}

/*
 * get the geometry buffer
 *
 */
void
D3plotBuffer::read_geometryBuffer()
{

  this->current_buffer = D3plotBuffer::get_bufferFromFile(d3plots[0]);
};

/*
 * free the geometry buffer
 *
 */
void
D3plotBuffer::free_geometryBuffer(){};

/*
 * Get the part buffer
 *
 */
void
D3plotBuffer::read_partBuffer(){};

/*
 * free the part buffer
 *
 */
void
D3plotBuffer::free_partBuffer(){};

/*
 * init the reading of the states
 *
 */
void
D3plotBuffer::init_nextState()
{

  if (this->current_buffer.size() == 0) {
    this->current_buffer = D3plotBuffer::get_bufferFromFile(d3plots[0]);
  }

// empty remaining data (prevents memory leak)
#ifdef QD_DEBUG
  std::cout << "Emptying previous IO-Buffers" << endl;
#endif
  while (state_buffers.size() != 0) {
    state_buffers.back().get();
    state_buffers.pop_back();
  }

  // preload buffers
  for (size_t iFile = d3plots.size() - 1; iFile > 0; --iFile) {
    state_buffers.push_back(
      std::async(D3plotBuffer::get_bufferFromFile, d3plots[iFile]));
  }
}

/*
 * Get the next state buffer
 *
 */
void
D3plotBuffer::read_nextState()
{

  // Do not load next buffer in case of first file
  // It will be read if the end marker is hit anyways.
  // Dont ask me why LS-DYNA is so complex ...
  if (iStateFile == 0) {
    iStateFile++;
    return;
  }

  if (iStateFile >= d3plots.size()) {
    throw(std::invalid_argument("There are no more state-files to be read."));
  }

#ifdef QD_DEBUG
  std::cout << "Loading state-file:" << d3plots[iStateFile] << endl;
#endif

  this->current_buffer = state_buffers.back().get();
  state_buffers.pop_back();
  iStateFile++;
  return;
}

/*
 * rewind the state reading.
 *
 */
void
D3plotBuffer::rewind_nextState()
{
  iStateFile = 0;
  this->current_buffer = D3plotBuffer::get_bufferFromFile(d3plots[0]);
}

/*
 * check if there is a next state
 *
 */
bool
D3plotBuffer::has_nextState()
{
  if (state_buffers.size() > 0)
    return true;
  return false;
}

/*
 * end the reading of states
 *
 */
void
D3plotBuffer::end_nextState()
{

  this->current_buffer.clear();
}

/*
 * Close the file ... just releases buffer.
 */
void
D3plotBuffer::finish_reading()
{

  this->current_buffer.clear();
}

} // namespace qd