
#include <sstream>
#include <string.h>
#include <iostream>
#include <bitset>
#include "../utility/FileUtility.h"
#include "D3plotBuffer.h"

/*
 * Constructor
 */
D3plotBuffer::D3plotBuffer(string _d3plot_path, int _wordSize){

  // Init vars
  iStateFile = 0;
  current_buffer = NULL;
  wordSize = 4;

  // Check File
  if(!FileUtility::check_ExistanceAndAccess(_d3plot_path)){
    throw("File \"" + _d3plot_path + "\" does not exist or is locked.");
  }

  this->d3plots = FileUtility::globVector(_d3plot_path+"*");

  if(d3plots.size() < 1)
    throw("No D3plot result file could be found with the given path:"+_d3plot_path);

  this->wordSize = _wordSize;
}


/*
 * Destructor
 */
D3plotBuffer::~D3plotBuffer(){

  if(this->current_buffer != NULL){
    delete[] current_buffer;
    this->current_buffer = NULL;
  }

}


/*
 * Get a char* byte buffer from the given file.
 *
 */
void D3plotBuffer::get_bufferFromFile(string filepath){

  if(this->current_buffer != NULL){
    delete[] this->current_buffer;
    this->current_buffer = NULL;
  }

  // Read data into buffer
  ifstream fStream;
  fStream.open(filepath.c_str(), ios::binary | ios::in);
  fStream.seekg(0,ios::end);
  this->bufferSize = fStream.tellg();
  fStream.seekg (0, ios::beg);
  //cout << "Filesize: " << *bufferSize << endl; // DEBUG
  this->current_buffer = new char [this->bufferSize];
  fStream.read (this->current_buffer, this->bufferSize);
  fStream.close();
}


/*
 * get the geometry buffer
 *
 */
void D3plotBuffer::read_geometryBuffer(){
  this->get_bufferFromFile(d3plots[0]);
};


/*
 * free the geometry buffer
 *
 */
void D3plotBuffer::free_geometryBuffer(){};


/*
 * Get the part buffer
 *
 */
void D3plotBuffer::read_partBuffer(){};


/*
 * free the part buffer
 *
 */
void D3plotBuffer::free_partBuffer(){};


/*
 * init the reading of the states
 *
 */
void D3plotBuffer::init_nextState(){

  if(this->current_buffer == NULL){
    this->get_bufferFromFile(d3plots[0]);
  }

}


/*
 * Get the next state buffer
 *
 */
void D3plotBuffer::read_nextState(){

  if(iStateFile == 0){
    iStateFile++;
    return;
  }

  if(iStateFile < d3plots.size()){
    #ifdef CD_DEBUG
    cout << "Loading state-file:" << d3plots[iStateFile] << endl;
    #endif
    this->get_bufferFromFile(d3plots[iStateFile]);
    iStateFile++;
    return;
  }

  throw("There are no more state-files to be read.");
}


/*
 * rewind the state reading.
 *
 */
void D3plotBuffer::rewind_nextState(){

  iStateFile = 0;
  this->get_bufferFromFile(d3plots[0]);

}


/*
 * check if there is a next state
 *
 */
bool D3plotBuffer::has_nextState(){
  if(iStateFile < d3plots.size())
    return true;
  return false;
}


/*
 * end the reading of states
 *
 */
void D3plotBuffer::end_nextState(){

  if(this->current_buffer != NULL){
    delete[] current_buffer;
    this->current_buffer = NULL;
  }
}

/*
 * Close the file ... just releases buffer.
 */
void D3plotBuffer::finish_reading(){

  if(this->current_buffer != NULL){
    delete[] current_buffer;
    this->current_buffer = NULL;
  }

}

/*
 * read an int from the current buffer
 */
int D3plotBuffer::read_int(int iWord){
  //if(this->bufferSize <= iWord*this->wordSize){
  //  throw("read_int tries to read beyond the buffer size.");
  //}

  // BIG ENDIAN ?
  // SMALL ENDIAN ?
  int start=iWord*this->wordSize;
  return (((current_buffer[start + 3] & 0xff) << 24)
          | ((current_buffer[ start+ 2] & 0xff) << 16)
          | ((current_buffer[start + 1] & 0xff) << 8)
          | ((current_buffer[start + 0] & 0xff)));
}


/*
 * read a float from the current buffer
 */
float D3plotBuffer::read_float(int iWord){
  //if(this->bufferSize <= iWord*this->wordSize){
  //  throw("read_float tries to read beyond the buffer size.");
  //}
  float ret;
  memcpy(&ret, &current_buffer[iWord*this->wordSize], sizeof(ret));
  return ret;
  //return (float) this->buffer[iWord*this->wordSize];
}


/*
 * read a string from the current buffer
 */
string D3plotBuffer::read_str(int iWord,int wordLength){
  //if(this->bufferSize <= (iWord+wordLength)*this->wordSize){
  //  throw("read_str tries to read beyond the buffer size.");
  stringstream res;
  for(int ii=iWord*this->wordSize;ii<(iWord+wordLength)*this->wordSize;ii++){
    res << char(bitset<8>(this->current_buffer[ii]).to_ulong());
  }

  return res.str();
}
