
#ifndef ABSTRACTBUFFER_HPP
#define ABSTRACTBUFFER_HPP

#include <bitset>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstring> // std::memcopy
#include <vector>

namespace qd {

class AbstractBuffer
{

protected:
  int32_t wordSize;
  std::vector<char> current_buffer;

public:
  // Standard
  AbstractBuffer(int32_t _wordSize)
    : wordSize(_wordSize){};
  virtual ~AbstractBuffer(){};
  // Geometry
  virtual void read_geometryBuffer() = 0;
  virtual void free_geometryBuffer() = 0;
  // Parts
  virtual void read_partBuffer() = 0;
  virtual void free_partBuffer() = 0;
  // States
  virtual void init_nextState() = 0;
  virtual void read_nextState() = 0;
  virtual bool has_nextState() = 0;
  virtual void rewind_nextState() = 0;
  virtual void end_nextState() = 0;
  // Close
  virtual void finish_reading() = 0;

  // Vars
  inline int32_t read_int(int32_t _iWord);
  inline float read_float(int32_t _iWord);
  inline void read_float_array(int32_t _iWord,
                               int32_t _length,
                               std::vector<float>& _buffer);
  inline std::string read_str(int32_t _iWord, int32_t _length);
};

/*
 * read an int32_t from the current buffer
 */
int32_t
AbstractBuffer::read_int(int32_t iWord)
{

#ifdef QD_DEBUG
  if (this->current_buffer.capacity() <= iWord * this->wordSize)
    throw(
      std::invalid_argument("read_int tries to read beyond the buffer size."));
#endif

  // BIG ENDIAN ?
  // SMALL ENDIAN ?
  int32_t start = iWord * this->wordSize;
  return (((current_buffer[start + 3] & 0xff) << 24) |
          ((current_buffer[start + 2] & 0xff) << 16) |
          ((current_buffer[start + 1] & 0xff) << 8) |
          ((current_buffer[start + 0] & 0xff)));
  // int32_t header = *reinterpret_cast<const int32_t*>(&soundFileDataVec[4]);
}

/*
 * read a float from the current buffer
 */
float
AbstractBuffer::read_float(int32_t iWord)
{

#ifdef QD_DEBUG
  if (this->current_buffer.capacity() <= iWord * this->wordSize)
    throw(
      std::invalid_argument("read_int tries to read beyond the buffer size."));
#endif

  float ret;
  std::memcpy(&ret, &current_buffer[iWord * this->wordSize], sizeof(ret));
  // return *reinterpret_cast<const
  // float*>(&current_buffer[iWord*this->wordSize]);
  return ret;
  // return (float) this->buffer[iWord*this->wordSize];
}

/*
 * Read a float array into an allocated buffer
 */
void
AbstractBuffer::read_float_array(int32_t _iWord,
                                 int32_t _length,
                                 std::vector<float>& _buffer)
{

#ifdef QD_DEBUG
  if (_buffer.capacity() < _length)
    throw(std::invalid_argument(
      "Can not read float array, container capacity too small."));
  if (this->current_buffer.capacity() <= (_iWord + _length) * this->wordSize)
    throw(std::invalid_argument(
      "read_float_array tries to read beyond the buffer size."));
#endif

  /*
  BUGGY
  int32_t pos = _iWord*this->wordSize;
  std::copy(&current_buffer[pos],
            &current_buffer[pos]+_length*sizeof(float),
            &_buffer[0]);
  */
  std::memcpy(&_buffer[0],
              &current_buffer[_iWord * this->wordSize],
              sizeof(float) * _length);
}

/*
 * read a string from the current buffer
 */
std::string
AbstractBuffer::read_str(int32_t iWord, int32_t wordLength)
{
  // if(this->bufferSize <= (iWord+wordLength)*this->wordSize){
  //  throw("read_str tries to read beyond the buffer size.");
  std::stringstream res;
  for (int32_t ii = iWord * this->wordSize;
       ii < (iWord + wordLength) * this->wordSize;
       ii++) {
    res << char(std::bitset<8>(this->current_buffer[ii]).to_ulong());
  }

  return res.str();
}

} // namespace qd

#endif
