

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#include "dyna_cpp/dyna/d3plot/D3plotBuffer.hpp"
#include "dyna_cpp/utility/FileUtility.hpp"

namespace qd {

/*
 * Constructor
 */
D3plotBuffer::D3plotBuffer(std::string _d3plot_path, int32_t word_size)
  : AbstractBuffer(word_size)
  , iStateFile(0)
{

  // Check File
  if (!check_ExistanceAndAccess(_d3plot_path)) {
    throw(std::invalid_argument("File \"" + _d3plot_path +
                                "\" does not exist or is locked."));
  }

  _d3plots = find_dyna_result_files(_d3plot_path);
#ifdef QD_DEBUG
  std::cout << "Found result files:" << std::endl;
  for (size_t ii = 0; ii < _d3plots.size(); ++ii) {
    std::cout << _d3plots[ii] << std::endl;
  }
  std::cout << "End of file list." << std::endl;
#endif

  if (_d3plots.size() < 1)
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
  /*
  while (file_buffer_q.size() != 0) {
    file_buffer_q.back().get();
    file_buffer_q.pop_back();
  }
  */
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
  std::streamoff bufferSize = fStream.tellg();
  fStream.seekg(0, std::ios::beg);
  state_buffer.resize(bufferSize);
  fStream.read(&state_buffer[0], bufferSize);
  fStream.close();

#ifdef QD_DEBUG
  std::cout << "Loaded file: " << filepath << '\n';
#endif

  return std::move(state_buffer);
}

/*
 * get the geometry buffer
 *
 */
void
D3plotBuffer::read_geometryBuffer()
{
  if (_current_buffer.size() == 0)
    _current_buffer = D3plotBuffer::get_bufferFromFile(_d3plots[0]);
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
  iStateFile = 0;

  if (_current_buffer.size() == 0)
    _current_buffer = D3plotBuffer::get_bufferFromFile(_d3plots[0]);

// empty remaining data (prevents memory leak)
#ifdef QD_DEBUG
  std::cout << "Emptying previous IO-Buffers" << std::endl;
#endif
  while (_file_buffer_q.size() != 0) {
    _file_buffer_q.back().get();
    _file_buffer_q.pop_back();
  }

  constexpr size_t n_threads = 1;

  _work_queue.reset();
  for (size_t iFile = 1; iFile < _d3plots.size(); ++iFile) {
    _file_buffer_q.push_back(
      _work_queue.submit(D3plotBuffer::get_bufferFromFile, _d3plots[iFile]));
  }
  _work_queue.init_workers(n_threads);

  // preload buffers
  // for (size_t iFile = _d3plots.size() - 1; iFile > 0; --iFile)
  // {
  //   file_buffer_q.push_back(
  //     std::async();
  // }
}

/*
 * Get the next state buffer
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

  if (iStateFile >= _d3plots.size()) {
    throw(std::runtime_error("There are no more state-files to be read."));
  }

#ifdef QD_DEBUG
  std::cout << "Loading state-file:" << _d3plots[iStateFile] << std::endl;
#endif

  _current_buffer = _file_buffer_q.front().get();
  _file_buffer_q.pop_front();

  iStateFile++;
}

/*
 * rewind the state reading.
 *
 */
void
D3plotBuffer::rewind_nextState()
{
  this->init_nextState();
}

/*
 * check if there is a next state
 *
 */
bool
D3plotBuffer::has_nextState()
{
  if (iStateFile == 0)
    return true;

  if (_file_buffer_q.size() > 0)
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
  _current_buffer.clear();
  _file_buffer_q.clear();
  _work_queue.abort();
}

/*
 * Close the file ... releases buffers.
 */
void
D3plotBuffer::finish_reading()
{
  _current_buffer.clear();
  _file_buffer_q.clear();
  _work_queue.abort();
}

} // namespace qd