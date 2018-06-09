
#ifndef ABSTRACTBUFFER_HPP
#define ABSTRACTBUFFER_HPP

#include <bitset>
#include <cstdint>
#include <cstring> // std::memcopy
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qd {

class AbstractBuffer
{

protected:
  int32_t _word_size;
  std::vector<char> _current_buffer;

public:
  // Standard
  AbstractBuffer(int32_t word_size)
    : _word_size(word_size){};
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
  inline int32_t read_int(int32_t _iWord) const;
  inline float read_float(int32_t _iWord) const;
  inline void read_float_array(int32_t _iWord,
                               int32_t _length,
                               std::vector<float>& _buffer) const;
  inline std::string read_str(int32_t _iWord, int32_t _length) const;
  template<typename T>
  void read_array(int32_t _iWord,
                  int32_t _length,
                  std::vector<T>& _buffer,
                  size_t _buffer_beginning = 0) const;
};

/*
 * read an int32_t from the current buffer
 */
int32_t
AbstractBuffer::read_int(int32_t iWord) const
{

#ifdef QD_DEBUG
  if (this->_current_buffer.capacity() <=
      static_cast<size_t>(iWord * this->_word_size))
    throw(
      std::invalid_argument("read_int tries to read beyond the buffer size."));
#endif

  // BIG ENDIAN ?
  // SMALL ENDIAN ?
  int32_t start = iWord * this->_word_size;
  return (((_current_buffer[start + 3] & 0xff) << 24) |
          ((_current_buffer[start + 2] & 0xff) << 16) |
          ((_current_buffer[start + 1] & 0xff) << 8) |
          ((_current_buffer[start + 0] & 0xff)));
  // int32_t header = *reinterpret_cast<const int32_t*>(&soundFileDataVec[4]);
}

/*
 * read a float from the current buffer
 */
float
AbstractBuffer::read_float(int32_t iWord) const
{

#ifdef QD_DEBUG
  if (this->_current_buffer.capacity() <=
      static_cast<size_t>(iWord * this->_word_size))
    throw(
      std::invalid_argument("read_int tries to read beyond the buffer size."));
#endif

  float ret;
  std::memcpy(&ret, &_current_buffer[iWord * this->_word_size], sizeof(ret));
  // return *reinterpret_cast<const
  // float*>(&_current_buffer[iWord*this->_word_size]);
  return ret;
  // return (float) this->buffer[iWord*this->_word_size];
}

template<typename T>
void
AbstractBuffer::read_array(int32_t _iWord,
                           int32_t _length,
                           std::vector<T>& _buffer,
                           size_t _buffer_beginning) const
{

#ifdef QD_DEBUG
  if (_buffer.capacity() < static_cast<size_t>(_length))
    throw(std::invalid_argument(
      "Can not read array, container capacity too small."));
  if (this->_current_buffer.capacity() <=
      static_cast<size_t>((_iWord + _length) * this->_word_size))
    throw(std::invalid_argument(
      "AbstractBuffer::read_array tries to read beyond the buffer size."));
#endif

  /*
  BUGGY
  int32_t pos = _iWord*this->_word_size;
  std::copy(&_current_buffer[pos],
            &_current_buffer[pos]+_length*sizeof(float),
            &_buffer[0]);
  */
  std::memcpy(&_buffer[_buffer_beginning],
              &_current_buffer[_iWord * this->_word_size],
              sizeof(T) * _length);
}

/*
 * Read a float array into an allocated buffer
 */
void
AbstractBuffer::read_float_array(int32_t _iWord,
                                 int32_t _length,
                                 std::vector<float>& _buffer) const
{

#ifdef QD_DEBUG
  if (_buffer.capacity() < static_cast<size_t>(_length))
    throw(std::invalid_argument(
      "Can not read float array, container capacity too small."));
  if (this->_current_buffer.capacity() <=
      static_cast<size_t>((_iWord + _length) * this->_word_size))
    throw(std::invalid_argument(
      "read_float_array tries to read beyond the buffer size."));
#endif

  /*
  BUGGY
  int32_t pos = _iWord*this->_word_size;
  std::copy(&_current_buffer[pos],
            &_current_buffer[pos]+_length*sizeof(float),
            &_buffer[0]);
  */
  std::memcpy(&_buffer[0],
              &_current_buffer[_iWord * this->_word_size],
              sizeof(float) * _length);
}

/*
 * read a string from the current buffer
 */
std::string
AbstractBuffer::read_str(int32_t iWord, int32_t wordLength) const
{
#ifdef QD_DEBUG
  if (this->_current_buffer.capacity() <=
      static_cast<size_t>((iWord + wordLength) * this->_word_size))
    throw(
      std::invalid_argument("read_str tries to read beyond the buffer size."));
#endif

  std::stringstream res;
  for (int32_t ii = iWord * this->_word_size;
       ii < (iWord + wordLength) * this->_word_size;
       ii++) {
    res << char(std::bitset<8>(this->_current_buffer[ii]).to_ulong());
  }

  return res.str();
}

} // namespace qd

#endif
