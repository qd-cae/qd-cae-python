
#ifndef ABSTRACTBUFFER_HPP
#define ABSTRACTBUFFER_HPP

#include <sstream>
#include <bitset>
#include <string>

class AbstractBuffer{

protected:
  int wordSize;
  std::vector<char> current_buffer;

public:
  // Standard
  AbstractBuffer(int _wordSize) 
    : wordSize(_wordSize) {};
  virtual ~AbstractBuffer(){};
  // Geometry
  virtual void read_geometryBuffer()=0;
  virtual void free_geometryBuffer()=0;
  // Parts
  virtual void read_partBuffer()=0;
  virtual void free_partBuffer()=0;
  // States
  virtual void init_nextState()=0;
  virtual void read_nextState()=0;
  virtual bool has_nextState()=0;
  virtual void rewind_nextState()=0;
  virtual void end_nextState()=0;
  // Close
  virtual void finish_reading()=0;

  // Vars
  inline int read_int(int _iWord);
  inline float read_float(int _iWord);
  inline void read_float_array(int _iWord, 
                               int _length, 
                               std::vector<float>& _buffer);
  inline std::string read_str(int _iWord,int _length);

};


/*
 * read an int from the current buffer
 */
int AbstractBuffer::read_int(int iWord){
  
  #ifdef QD_DEBUG
  if(this->current_buffer.capacity() <= iWord*this->wordSize)
    throw(std::string("read_int tries to read beyond the buffer size."));
  #endif

  // BIG ENDIAN ?
  // SMALL ENDIAN ?
  int start=iWord*this->wordSize;
  return (((current_buffer[start + 3] & 0xff) << 24)
          | ((current_buffer[ start+ 2] & 0xff) << 16)
          | ((current_buffer[start + 1] & 0xff) << 8)
          | ((current_buffer[start + 0] & 0xff)));
  //int header = *reinterpret_cast<const int*>(&soundFileDataVec[4]);

}


/*
 * read a float from the current buffer
 */
float AbstractBuffer::read_float(int iWord){
  
  #ifdef QD_DEBUG
  if(this->current_buffer.capacity() <= iWord*this->wordSize)
    throw(std::string("read_int tries to read beyond the buffer size."));
  #endif

  float ret;
  memcpy(&ret, &current_buffer[iWord*this->wordSize], sizeof(ret));
  //return *reinterpret_cast<const float*>(&current_buffer[iWord*this->wordSize]);
  return ret;
  //return (float) this->buffer[iWord*this->wordSize];
}


/*
 * Read a float array into an allocated buffer
 */
void AbstractBuffer::read_float_array(int _iWord, 
                                    int _length, 
                                    std::vector<float>& _buffer){

   #ifdef QD_DEBUG
   if( _buffer.capacity() < _length )
    throw(std::string("Can not read float array, container capacity too small."));
   if(this->current_buffer.capacity() <= (_iWord+_length)*this->wordSize)
    throw(std::string("read_float_array tries to read beyond the buffer size."));
   #endif
   
   /*
   BUGGY
   int pos = _iWord*this->wordSize;
   std::copy(&current_buffer[pos], 
             &current_buffer[pos]+_length*sizeof(float), 
             &_buffer[0]);
   */
   memcpy(&_buffer[0], &current_buffer[_iWord*this->wordSize], sizeof(float)*_length);
}


/*
 * read a string from the current buffer
 */
std::string AbstractBuffer::read_str(int iWord,int wordLength){
  //if(this->bufferSize <= (iWord+wordLength)*this->wordSize){
  //  throw("read_str tries to read beyond the buffer size.");
  std::stringstream res;
  for(int ii=iWord*this->wordSize;ii<(iWord+wordLength)*this->wordSize;ii++){
    res << char(std::bitset<8>(this->current_buffer[ii]).to_ulong());
  }

  return res.str();
}


#endif
