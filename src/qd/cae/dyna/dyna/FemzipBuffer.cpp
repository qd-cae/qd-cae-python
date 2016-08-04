
extern "C"
{
#include "femzip.h"
#include <stdio.h>
}
#include "FemzipBuffer.h"
#include "../utility/FileUtility.h"
#include <iostream>
#include <bitset>
#include <sstream>

/*
 * Constructor
 */
FemzipBuffer::FemzipBuffer(string _filepath){

  // Init vars
  this->filepath = _filepath;
  if(!FileUtility::check_ExistanceAndAccess(this->filepath)){
    throw("File \"" + this->filepath + "\" does not exist or is locked.");
  }
  this->init_vars();
  
  // version check
  float unzipversion = 0.;
	float fileunzipversion = 0.;

  femunziplib_version(&unzipversion);
  femunziplib_version_file((char*) this->filepath.c_str(), &this->filetype, &fileunzipversion,&this->ier);
  this->check_ier("Femzip Error during femunziplib_version_file.");

  if(unzipversion<fileunzipversion){
    throw("Femzip version older than file version.");
  }

	/* SIZES */
	get_Size((char*) this->filepath.c_str(), this->filetype, this->adjust, &this->size_geo,
		&this->size_state, &this->size_disp, &this->size_activity,
		&this->size_post, &this->ier);
	this->check_ier("Femzip Error during reading of sizes.");

}


/*
 * Destructor
 */
FemzipBuffer::~FemzipBuffer(){

  if(this->current_buffer != NULL){
    delete[] this->current_buffer;
    this->current_buffer = NULL;
  }
}

/*
 * Initialize the variables for the class.
 */
void FemzipBuffer::init_vars(){
   
  this->wordSize = 4; // byte
  this->current_buffer = NULL;

  // general
  this->filetype = 1;
  this->ier = 0;
  this->pos = 0;
  // Sizing
  this->size_geo = 0;
  this->size_state = 0;
  this->size_disp = 0;
  this->size_activity = 0;
  this->size_post = 0;
  this->size_titles = 0;
  // States
  this->iTimeStep = 1; // ... why
  this->nTimeStep = 0;
  this->size_times = 0;
  // config
  this->adjust = 5;
   
}

/*
 * Read the geometry buffer.
 */
void FemzipBuffer::read_geometryBuffer(){

  string nonsense_s = "nonsense";
  char* nonsense = (char*) nonsense_s.c_str();
  char* argv[] = { nonsense, (char*) this->filepath.c_str() };
  int p1[1000];
  int p2[1000];
  int l1 = 0;
  int l2 = 0;
  wrapinput(2, argv, p1, p2, &l1, &l2);

  if(this->current_buffer != NULL){
    delete[] this->current_buffer;
    this->current_buffer = NULL;
  }

  this->current_buffer = new char[sizeof(int)*this->size_geo];
  geometry_read(p1, p2, &l1, &l2, &this->ier, &this->pos, (int*) this->current_buffer, &this->size_geo);
  this->check_ier("Femzip Error while reading geometry.");

}


/*
 * Free the geometry buffer.
 */
void FemzipBuffer::free_geometryBuffer(){

  if(this->current_buffer != NULL){
    delete[] this->current_buffer;
    this->current_buffer = NULL;
  }

}


/*
 * Read the part buffer.
 */
void FemzipBuffer::read_partBuffer(){

  if(this->current_buffer != NULL){
    delete[] this->current_buffer;
    this->current_buffer = NULL;
  }

  this->pos = 0;
  part_titles_read(&this->ier, &this->pos, (int*) this->current_buffer, &this->size_titles);
  this->current_buffer = new char[sizeof(int)*this->size_titles];
  this->pos = 0;
  part_titles_read(&this->ier, &this->pos, (int*) this->current_buffer, &this->size_titles);
	check_ier("Femzip Error during part_titles_read.");

}


/*
 * Free the part buffer.
 */
void FemzipBuffer::free_partBuffer(){

  if(this->current_buffer != NULL){
    delete[] this->current_buffer;
    this->current_buffer = NULL;
  }

}


/*
 * Init the state reading.
 */
void FemzipBuffer::init_nextState(){

  this->iTimeStep = 1;
  int retry = 0;
  int size_times = 2000;
  retry: this->timese = new float[size_times];
	this->pos = 0;
	ctimes_read(&this->ier, &this->pos, &this->nTimeStep, this->timese, &size_times);
	if (this->ier == 9)
	{
		if (retry < 1) {
			retry++;
			goto retry;
		}
    if(this->timese != NULL){
      delete[] this->timese;
      this->timese = NULL;
    }
    throw("Femzip Buffer Error: size for the states buffer is to small.");
	}

  if(this->timese != NULL){
    delete[] this->timese;
    this->timese = NULL;
  }
	check_ier("Femzip Error during ctimes_read.");

  this->timese = NULL;
}


/*
 * Init the state reading.
 */
void FemzipBuffer::read_nextState(){

  // Discard previous one
  if(this->current_buffer != NULL){
    delete[] this->current_buffer;
    this->current_buffer = NULL;
  }
  
  #ifdef CD_DEBUG
  cout << "Loading state: " << this->iTimeStep << "/" << this->nTimeStep << endl;
  #endif

  this->current_buffer = new char[sizeof(int)*this->size_state];
  this->pos = 0;
  states_read(&this->ier, &this->pos, &this->iTimeStep,(int*) this->current_buffer, &this->size_state);
  check_ier("Femzip Error during states_read.");
  this->iTimeStep++;

}


/*
 * Is there another state?
 */
bool FemzipBuffer::has_nextState(){
  return this->iTimeStep <= this->nTimeStep;
}


/*
 * Is there another state?
 */
void FemzipBuffer::rewind_nextState(){

  this->iTimeStep = 1;
  if(this->current_buffer != NULL){
    delete[] this->current_buffer;
    this->current_buffer = NULL;
  }

  // Size
  this->pos =0;
	get_Size((char*) this->filepath.c_str(), this->filetype, this->adjust, &this->size_geo,
		&this->size_state, &this->size_disp, &this->size_activity,
		&this->size_post, &this->ier);
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
void FemzipBuffer::end_nextState(){

  states_close(&this->ier, &this->pos, (int*) this->current_buffer, &this->size_geo);
  if(this->current_buffer != NULL){
    delete[] this->current_buffer;
    this->current_buffer = NULL;
  }
  
  close_read(&this->ier);
  this->check_ier("Femzip Error during closing of file.");

}

/*
 * Close the file.
 */ 
void FemzipBuffer::finish_reading(){

   close_read(&this->ier);
   this->check_ier("Femzip Error during closing of file.");
 
}

/*
 * Check for error
 */
void FemzipBuffer::check_ier(string message){
  if(this->ier != 0){
    if(this->current_buffer != NULL){
      delete[] this->current_buffer;
      this->current_buffer = NULL;
    }
    throw(message);
  }
}



/*
 * read an int from the current buffer
 */
int FemzipBuffer::read_int(int iWord){
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
float FemzipBuffer::read_float(int iWord){
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
string FemzipBuffer::read_str(int iWord,int wordLength){
  //if(this->bufferSize <= (iWord+wordLength)*this->wordSize){
  //  throw("read_str tries to read beyond the buffer size.");
  stringstream res;
  for(int ii=iWord*this->wordSize;ii<(iWord+wordLength)*this->wordSize;ii++){
    res << char(bitset<8>(this->current_buffer[ii]).to_ulong());
  }

  return res.str();
}
