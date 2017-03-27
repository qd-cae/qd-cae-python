
#include <cmath>
#include <string>
#include "dyna_cpp/dyna/D3plot.hpp"
#include "dyna_cpp/dyna/D3plotBuffer.hpp"
#include "dyna_cpp/db/Node.hpp"
#include "dyna_cpp/db/Part.hpp"
#include "dyna_cpp/db/Element.hpp"
#include "dyna_cpp/db/DB_Nodes.hpp"
#include "dyna_cpp/db/DB_Parts.hpp"
#include "dyna_cpp/db/DB_Elements.hpp"
#include "dyna_cpp/utility/TextUtility.hpp"
#include "dyna_cpp/utility/FileUtility.hpp"
#include "dyna_cpp/utility/MathUtility.hpp"

#include <set>

#ifdef QD_USE_FEMZIP
#include "FemzipBuffer.hpp"
#endif

using namespace std;

/** Constructor for a D3plot.
 * @param string filepath : path to the d3plot file
 * @param vector<string> state_variables : which state variables to read
 * @param bool use_femzip : set to true if your d3plot was femzipped
 */
D3plot::D3plot (string _filename, vector<string> _state_variables, bool _useFemzip)
: FEMFile(_filename) {

   // standard vars
   this->init_vars();

  // maybe make it as argument in future
  const int bytesPerWord = 4; // future argument for DP
  this->useFemzip = _useFemzip;

  // Create Buffer
  #ifdef QD_DEBUG
  cout << "Reading d3plot ... " << endl;
  #endif

  #ifdef QD_USE_FEMZIP
  if(this->useFemzip){
    this->buffer = new FemzipBuffer(this->get_filepath());
  } else {
    this->buffer = new D3plotBuffer(this->get_filepath(),bytesPerWord);
  }
  #else
  if(this->useFemzip){
	  throw(string("d3plot.cpp was compiled without femzip support."));
  }
  this->buffer = new D3plotBuffer(this->get_filepath(),bytesPerWord);
  #endif

  this->buffer->read_geometryBuffer(); // deallocated in read_geometry

  // Header + Geometry
  this->read_header();
  this->read_matsection();
  this->read_geometry();

  // States
  //
  // This routine must run through, even though no variables might be read.
  // This is due to the fact, that femzip must read the states before
  // closing the file. It is not possible to leave this out.
  this->read_states(_state_variables);

}


/*
 * Destructor
 *
 */
D3plot::~D3plot(){

  if (this->buffer != nullptr) {
    delete this->buffer;
  }

}


/*
 * Initialize the object vars.
 */
void D3plot::init_vars(){

  this->dyna_ndim = -1; // dimension parameter
  this->dyna_numnp = -1; // number of nodes
  this->dyna_mdlopt = -1; // describes element deletion
  this->dyna_mattyp = -1; // material types section is read

  this->dyna_nglbv = -1; // global vars per timestep

  this->dyna_nel2 = -1; // #elements with 2 nodes (beams)
  this->dyna_nel4 = -1; // #elements with 4 nodes (shells)
  this->dyna_nel48 = -1; // # 8 node shell elements?!?!?!
  this->dyna_nel8 = -1; // #elements with 8 nodes (solids)
  this->dyna_nel20 = -1; // # 20 node solid elements
  this->dyna_nelth = -1; // #thshells

  this->dyna_nmmat = -1;   // #mats
  this->dyna_nummat2 = -1; // #mats for 1d/2d/3d/th elems
  this->dyna_nummat4 = -1;
  this->dyna_nummat8 = -1;
  this->dyna_nummatth = -1;

  this->dyna_nv1d = -1; // #vars for 1d/2d/3d/th elems
  this->dyna_nv2d = -1;
  this->dyna_nv3d = -1;
  this->dyna_nv3dt = -1;

  this->dyna_maxint = -1; // #layers of integration points
  this->dyna_istrn = -1; // indicates whether strain was written
  this->dyna_neiph = -1; // extra variables for solids
  this->dyna_neips = -1; // extra variables for shells

  this->dyna_iu = -1; // Indicators for: disp/vel/accel/temp
  this->dyna_iv = -1;
  this->dyna_ia = -1;
  this->dyna_it = -1;
  this->dyna_idtdt = -1; // temp change rate, numnp vals after temps

  this->dyna_narbs = -1; // dunno ... seems important

  this->dyna_ioshl1 = -1; // 6 shell stresses
  this->dyna_ioshl2 = -1; // shell plastic strain
  this->dyna_ioshl3 = -1; // shell forces
  this->dyna_ioshl4 = -1; // thick,energy,2 extra

  this->dyna_extra = -1; // double header length indicator
  this->dyna_numprop = -1; // number of mats

  this->dyna_numrbe = -1; // number of rigid body shell elems
  
  // just for checks ... can not be handled.
  this->dyna_nmsph = -1; // #nodes of sph
  this->dyna_ngpsph = -1; // #mats of sph
  this->dyna_ialemat = -1; // # some ale stuff .. it's late ...

  // Own Variables
  this->nStates = -0;
  vector<float> timesteps;

  this->own_nel10 = false;
  this->own_external_numbers_I8 = false;

  this->wordPosition = 0;
  this->wordsToRead = 0;
  this->wordPositionStates = 0;

  this->useFemzip=false;
  this->femzip_state_offset=0;

  // Checks for already read variables
  this->plastic_strain_is_read = false;
  this->plastic_strain_read = 0;
  this->energy_is_read = false;
  this->energy_read = 0;
  this->strain_is_read = false;
  this->strain_read = 0;
  this->stress_is_read = false;
  this->stress_read = 0;
  this->stress_mises_is_read = false;
  this->stress_mises_read = 0;
  this->disp_is_read = false;
  this->disp_read = 0;
  this->acc_is_read = false;
  this->acc_read = 0;
  this->vel_is_read = false;
  this->vel_read = 0;

}


/*
 * Read the header data.
 *
 */
void D3plot::read_header(){

  #ifdef QD_DEBUG
  cout << "> HEADER " << endl;
  #endif

  int filetype = this->buffer->read_int(11);
  if(filetype > 1000){
    filetype -= 1000;
    own_external_numbers_I8 = true;
  }
  if( (filetype != 1) && (filetype != 5) ){
    throw("Wrong filetype "+to_string(this->buffer->read_int(11))+" != 1 (or 5) in header of d3plot. Your file might be in Double Precision or the endian of the file is not the endian of the machine.");
  }

  this->dyna_title = this->buffer->read_str(0,10);

  // BUGGY
  /*
  int timestamp = this->buffer->read_int(10);
  time_t timestamp2 = (time_t) timestamp;
  this->dyna_datetime = asctime(localtime(&timestamp2));
  cout << timestamp << endl;
  cout << this->dyna_datetime << endl;
  */

  this->dyna_ndim = this->buffer->read_int(15);
  this->dyna_mattyp = 0;
  if ( (this->dyna_ndim == 5) | (this->dyna_ndim == 7) ){
    // connectivities are unpacked?
    this->dyna_mattyp = 1;
    this->dyna_ndim = 3;
  } else if (this->dyna_ndim == 4) {
    // connectivities are unpacked?
    this->dyna_ndim = 3;
  } else if (this->dyna_ndim > 5) {
    throw(string("State data contains rigid road surface which can not be handled."));
  } else {
    throw(string("Invalid parameter dyna_ndim=")+to_string(this->dyna_ndim));
  }

  this->dyna_numnp = this->buffer->read_int(16);
  this->dyna_nglbv = this->buffer->read_int(18);

  this->dyna_iu = this->buffer->read_int(20);
  this->dyna_iv = this->buffer->read_int(21);
  this->dyna_ia = this->buffer->read_int(22);
  this->dyna_it = this->buffer->read_int(19);

  this->dyna_nel2 = this->buffer->read_int(28);
  this->dyna_nel4 = this->buffer->read_int(31);
  this->dyna_nel8 = this->buffer->read_int(23);
  this->dyna_nelth = this->buffer->read_int(40);
  this->dyna_nel48 = this->buffer->read_int(55);
  if (this->dyna_nel8 < 0){
    this->dyna_nel8 = abs(dyna_nel8);
    own_nel10 = true;
  }

  this->dyna_nmmat = this->buffer->read_int(51);
  this->dyna_nummat2 = this->buffer->read_int(29);
  this->dyna_nummat4 = this->buffer->read_int(32);
  this->dyna_nummat8 = this->buffer->read_int(24);
  this->dyna_nummatth = this->buffer->read_int(41);

  this->dyna_nv1d = this->buffer->read_int(30);
  this->dyna_nv2d = this->buffer->read_int(33);
  this->dyna_nv3d = this->buffer->read_int(27);
  this->dyna_nv3dt = this->buffer->read_int(42);

  this->dyna_neiph = this->buffer->read_int(34);
  this->dyna_neips = this->buffer->read_int(35);
  this->dyna_maxint = this->buffer->read_int(36);
  // this may be faulty in the documentation!
  // ... gotta check this
  // ... did it, now should be ok :)
  if (this->dyna_maxint >= 0){
    this->dyna_mdlopt = 0;
  } else if (this->dyna_maxint < 0){
    this->dyna_mdlopt = 1;
    this->dyna_maxint = abs(this->dyna_maxint);
  }
  if (this->dyna_maxint > 10000) {
    this->dyna_mdlopt = 2;
    this->dyna_maxint = abs(this->dyna_maxint)-10000;
  }

  this->dyna_narbs = this->buffer->read_int(39);

  this->dyna_ioshl1 = this->buffer->read_int(43);
  this->dyna_ioshl1 == 1000 ? this->dyna_ioshl1 = 1 : this->dyna_ioshl1 = 0;
  this->dyna_ioshl2 = this->buffer->read_int(44);
  this->dyna_ioshl2 == 1000 ? this->dyna_ioshl2 = 1 : this->dyna_ioshl2 = 0;
  this->dyna_ioshl3 = this->buffer->read_int(45);
  this->dyna_ioshl3 == 1000 ? this->dyna_ioshl3 = 1 : this->dyna_ioshl3 = 0;
  this->dyna_ioshl4 = this->buffer->read_int(46);
  this->dyna_ioshl4 == 1000 ? this->dyna_ioshl4 = 1 : this->dyna_ioshl4 = 0;

  this->dyna_idtdt = this->buffer->read_int(56);
  this->dyna_extra = this->buffer->read_int(57);

  // Just 4 checks
  this->dyna_nmsph = this->buffer->read_int(37);
  this->dyna_ngpsph = this->buffer->read_int(38);
  this->dyna_ialemat = this->buffer->read_int(47);

  // Header extra!
  if (this->dyna_extra > 0){
    this->dyna_nel20 = this->buffer->read_int(64);
  } else {
    this->dyna_nel20 = 0;
  }

  // istrn in idtdt
  if (this->dyna_idtdt > 100){

    this->dyna_istrn = this->dyna_idtdt % 10000;

  // istrn needs to be calculated
  } else {
    if (this->dyna_nv2d > 0) {
      if (this->dyna_nv2d - this->dyna_maxint *
          (6 * this->dyna_ioshl1 + this->dyna_ioshl2 + this->dyna_neips)
          + 8 * this->dyna_ioshl3 + 4 * this->dyna_ioshl4 > 1) {
            this->dyna_istrn = 1;
				  } else {
            this->dyna_istrn = 0;
          }
      }
  }

  #ifdef QD_DEBUG
  this->info();
  #endif

  /* === CHECKS === */
  // sph
  if((this->dyna_nmsph != 0) | (this->dyna_ngpsph != 0)) throw(string("SPH mats and elements can not be handled."));
  // ale
  if(this->dyna_ialemat != 0) throw(string("ALE can not be handled."));
  // thick shells not implemented
  if(this->dyna_nelth > 0) throw(string("Can not handle thick shell elements."));
  // no temps
  if(this->dyna_it != 0) throw(string("dyna_it != 0: Can not handle temperatures."));
  //
  if(own_external_numbers_I8) throw(string("Can not handle external ids with double length."));


  // update position
  if (this->dyna_extra > 0){
    wordPosition = 64*2; // header has 128 words
  } else {
    wordPosition = 64; // header has 64 words
  }

}

/** Print info about the data in the d3plot to the console.
 *
 */
void D3plot::info(){

  cout << "Title:  " << this->dyna_title << '\n';
  cout << "nNodes : " << this->dyna_numnp << '\n';
  cout << "nElem2 : " << this->dyna_nel2 << '\n';
  cout << "nElem4 : " << this->dyna_nel4 << '\n';
  cout << "nElem8 : " << this->dyna_nel8 << '\n';
  cout << "nElem20: " << this->dyna_nel20 << '\n';
  cout << "nElemTh: " << this->dyna_nelth << '\n';
  cout << "nElem48: " << this->dyna_nel48 << '\n';
  cout << "nMats-Solver: " << this->dyna_nmmat << '\n';
  cout << "nMats-Input : " << this->dyna_nummat2
                          +this->dyna_nummat4
                          +this->dyna_nummat8
                          +this->dyna_nummatth << '\n';
  cout << "nMat2 : " << this->dyna_nummat2 << '\n';
  cout << "nMat4 : " << this->dyna_nummat4 << '\n';
  cout << "nMat8 : " << this->dyna_nummat8 << '\n';
  cout << "nMatTh: " << this->dyna_nummatth << '\n';
  cout << "disp : " << this->dyna_iu << '\n';
  cout << "vel  : " << this->dyna_iv << '\n';
  cout << "accel: " << this->dyna_ia << '\n';
  cout << "temp : " << this->dyna_it << '\n';
  cout << "shell-stress: " << this->dyna_ioshl1 << '\n';
  cout << "shell-plstrn: " << this->dyna_ioshl2 << '\n';
  cout << "shell-forces: " << this->dyna_ioshl3 << '\n';
  cout << "shell-stuff : " << this->dyna_ioshl4 << '\n';
  cout << "shell-strn  : " << this->dyna_istrn << '\n';
  cout << "shell-nInteg: " << this->dyna_maxint << '\n';
  cout << "nVar1D: " << this->dyna_nv1d << '\n';
  cout << "nVar2D: " << this->dyna_nv2d << '\n';
  cout << "nVar3D: " << this->dyna_nv3d << '\n';
  cout << "state-globals: " << this->dyna_nglbv << endl;

}


/** Read the material section
 *
 * Does nothing if mattyp==0.
 */
void D3plot::read_matsection(){

  // Nothing to do
  if( this->dyna_mattyp==0 ){
    dyna_numrbe = 0;
    return;
  }

  this->dyna_numrbe = this->buffer->read_int(wordPosition); // rigid shells
  int tmp_nummat = this->buffer->read_int(wordPosition + 1);
  if( tmp_nummat != dyna_nmmat )
    throw(string("dyna_nmmat != nummat in matsection!"));

  int start = wordPosition+2;
  int end   = start+tmp_nummat;
  this->dyna_irbtyp.reserve(tmp_nummat);
  for(int iPosition=start; iPosition<end; ++iPosition){
    this->dyna_irbtyp.push_back( this->buffer->read_int(iPosition) );
  }

  this->wordPosition += 2 + tmp_nummat;

}


/** Read the geometry mesh (after the header)
 */
void D3plot::read_geometry(){

  #ifdef QD_DEBUG
  cout << "> GEOMETRY" << endl;
  #endif

  /* === NODES === */
  vector< vector<float> > buffer_nodes = this->read_geometry_nodes();

  /* === ELEMENTS === */
  // Order MATTERS, do not swap routines.

  // 8-Node Solids
  vector< vector<int> > buffer_elems8 = this->read_geometry_elem8();

  // 8-Node Thick Shells
  if(dyna_nelth > 0) wordPosition += 9*dyna_nelth;

  // 2-Node Beams
  vector< vector<int> > buffer_elems2 = this->read_geometry_elem2();

  // 4-Node Elements
  vector< vector<int> > buffer_elems4 = this->read_geometry_elem4();

  /* === NUMBERING === */
  vector< vector<int> > buffer_numbering = this->read_geometry_numbering();

  if(!isFileEnding(wordPosition)){
    throw(string("Anticipated file ending wrong in geometry section."));
  }
  wordPosition++;

  /* === PARTS === */
  // (+DB)
  this->buffer->free_geometryBuffer();
  this->buffer->read_partBuffer();
  if(this->useFemzip)
    wordPosition = 1; // don't ask me why not 0 ...
  this->read_geometry_parts();

  if(!isFileEnding(wordPosition)){
    throw(string("Anticipated file ending wrong in part section."));
  }

  this->buffer->free_partBuffer();

  /* ====== D A T A B A S E S ====== */

  // Nodes
  #ifdef QD_DEBUG
  cout << "Adding nodes ... ";
  #endif
  if(buffer_numbering[0].size() != buffer_nodes.size())
    throw(string("Buffer node-numbering and buffer-nodes have different sizes."));
  DB_Nodes *db_nodes = this->get_db_nodes();
  db_nodes->reserve(buffer_nodes.size());
  for(unsigned int ii=0; ii < buffer_nodes.size() ;ii++){
    db_nodes->add_node(buffer_numbering[0][ii],buffer_nodes[ii]);
  }
  #ifdef QD_DEBUG
  cout << this->get_db_nodes()->size() << " done." << endl;
  #endif
  
  // Beams
  #ifdef QD_DEBUG
  cout << "Adding beams ... ";
  #endif
  DB_Elements *db_elems = this->get_db_elements();
  db_elems->reserve(BEAM, buffer_elems2.size());
  for(size_t ii=0; ii < buffer_elems2.size() ;++ii){
    db_elems->add_element_byD3plot(BEAM,buffer_numbering[2][ii],buffer_elems2[ii]);
	if( (dyna_mattyp==1) && (this->dyna_irbtyp[ buffer_elems2[ii].back() ] == 20) ) {
    }
  }
  #ifdef QD_DEBUG
  cout << this->get_db_elements()->size(BEAM) << " done." << endl;
  #endif

  // Shells
  #ifdef QD_DEBUG
  cout << "Adding shells ... ";
  #endif
  size_t nRigidShells = 0;
  db_elems->reserve(SHELL, buffer_elems4.size());
  for(size_t ii=0; ii < buffer_elems4.size() ;++ii){
    Element *elem = db_elems->add_element_byD3plot(SHELL,buffer_numbering[3][ii],buffer_elems4[ii]);
    
    // check if rigid material, very complicated ...
    // this bug took me 3 Days! material indexes start again at 1, not 0 :(
    if( (dyna_mattyp==1) && (this->dyna_irbtyp[ buffer_elems4[ii].back()-1 ] == 20) ) { 
      elem->set_is_rigid(true);
      ++nRigidShells;
    }
  }
  #ifdef QD_DEBUG
  cout << this->get_db_elements()->size(SHELL) << " done." << endl;
  #endif
  if( nRigidShells != this->dyna_numrbe )
    throw(string("nRigidShells != numrbe: ")+to_string(nRigidShells)+" != "+to_string(this->dyna_numrbe));

  // Solids
  #ifdef QD_DEBUG
  cout << "Adding solids ... ";
  #endif
  db_elems->reserve(SOLID, buffer_elems8.size());
  for(size_t ii=0; ii < buffer_elems8.size() ;++ii){
    db_elems->add_element_byD3plot(SOLID, buffer_numbering[1][ii], buffer_elems8[ii]);
	if( (dyna_mattyp==1) && (this->dyna_irbtyp[ buffer_elems8[ii].back() ] == 20) ) {
    }
  }
  #ifdef QD_DEBUG
  cout << get_db_elements()->size(SOLID) << " done." << endl;
  #endif
  
}


/*
 * Read the nodes in the geometry section.
 *
 */
vector< vector<float> > D3plot::read_geometry_nodes(){

  #ifdef QD_DEBUG
  cout << "Reading nodes ... ";
  #endif

  wordsToRead = dyna_numnp*dyna_ndim;
  vector< vector<float> > buffer_nodes(dyna_numnp,vector<float>(3));

  size_t jj = 0;
  for (int ii=wordPosition;ii<wordPosition+wordsToRead; ii+=3, ++jj){
    
    buffer->read_float_array(ii, 3, buffer_nodes[jj]);
    /*
    buffer_nodes[jj][0] = buffer->read_float(ii);
    buffer_nodes[jj][1] = buffer->read_float(ii+1);
    buffer_nodes[jj][2] = buffer->read_float(ii+2);
    */

  }

  // Update word position
  wordPosition += wordsToRead;

  #ifdef QD_DEBUG
  cout << "done." << endl;
  #endif

  return buffer_nodes;

}


/*
 * Read the 8 noded elements in the geometry section.
 *
 */
vector< vector<int> > D3plot::read_geometry_elem8(){

  // Check
  if(dyna_nel8 == 0) return vector< vector<int> >();

  #ifdef QD_DEBUG
  cout << "Reading elems8 ... ";
  #endif

  // currently each element has 8 nodes-ids and 1 mat-id
  const int nVarsElem8 = 9;

  // allocate
  vector< vector<int> > buffer_elems8(dyna_nel8, vector<int>(nVarsElem8));

  wordsToRead = nVarsElem8*dyna_nel8;
  size_t iElement = 0;
  size_t iData = 0;
  // Loop over elements
  for (int ii=wordPosition;ii<wordPosition+wordsToRead;ii+=nVarsElem8){

    // Loop over element data
    iData = 0;
    for(int jj=ii;jj<ii+nVarsElem8;jj++){
      buffer_elems8[iElement][iData] = buffer->read_int(jj);
      iData++;
    }

    iElement++;
  }

  // Update word position
  wordPosition += wordsToRead;
  if(own_nel10) wordPosition += 2*dyna_nel8;

  #ifdef QD_DEBUG
  cout << "done." << endl;
  #endif

  return std::move(buffer_elems8);

}


/*
 * Read the 4 noded elements in the geometry section.
 *
 */
vector< vector<int> > D3plot::read_geometry_elem4(){

  // Check
  if(dyna_nel4 == 0) return vector< vector<int> >();

  #ifdef QD_DEBUG
  cout << "Reading elems4 ... ";
  #endif

  const int nVarsElem4 = 5;

  // allocate
  vector< vector<int> > buffer_elems4(dyna_nel4, vector<int>(nVarsElem4));

  wordsToRead = nVarsElem4*dyna_nel4;
  size_t iElement = 0;
  size_t iData = 0;
  // Loop over elements
  for (int ii=wordPosition; ii<wordPosition+wordsToRead; ii+=nVarsElem4){

    // Loop over element data
    iData = 0;
    for(int jj=ii; jj<ii+nVarsElem4; ++jj){
      buffer_elems4[iElement][iData] = buffer->read_int(jj);
      ++iData;
    }
    ++iElement;

  }

  // Update word position
  wordPosition += wordsToRead;

  #ifdef QD_DEBUG
  cout << "done." << endl;
  #endif

  return std::move(buffer_elems4);
}


/*
 * Read the 2 noded elements in the geometry section.
 *
 */
vector< vector<int> > D3plot::read_geometry_elem2(){

  // Check
  if(dyna_nel2 == 0) return vector< vector<int> >();

  #ifdef QD_DEBUG
  cout << "Reading elems2 ... ";
  #endif

  const int nVarsElem2 = 6;

  // allocate
  vector< vector<int> > buffer_elems2(dyna_nel2, vector<int>(3));

  wordsToRead = nVarsElem2*dyna_nel2;
  int iElement = 0;
  // Loop over elements
    for (int ii=wordPosition;ii<wordPosition+wordsToRead;ii+=nVarsElem2){

    // Loop over element data
    /*
    iData = 0;
    vector<int> elemData(nVarsElem2);
    for(int jj=ii;jj<ii+nVarsElem2;jj++){
      elemData[iData] = buffer->read_int(jj);
      iData++;
    }
    */
    buffer_elems2[iElement][0] = buffer->read_int(ii);
    buffer_elems2[iElement][1] = buffer->read_int(ii+1);
    buffer_elems2[iElement][2] = buffer->read_int(ii+5); // mat

    iElement++;
  }

  // Update word position
  wordPosition += wordsToRead;

  #ifdef QD_DEBUG
  cout << "done." << endl;
  #endif

  return std::move(buffer_elems2);
}


/*
 * Read the numbering of the data into a 2d-vector.
 * numbering[iCategory][iIndex]
 * category: nodes  = 0
 *           solids = 1
 *           beams  = 2
 *           shells = 3
 *           (tshells= 4 - not implemented)
 */
vector< vector<int> > D3plot::read_geometry_numbering(){

  // TODO
  // NARBS check wrong?!?!?!
  /*
  int nHeaderWords = 10;
  int kk = 10+dyna_numnp+dyna_nel2+dyna_nel4+dyna_nel8+dyna_nelth;
  if(dyna_narbs == kk){
    nHeaderWords = 10;
  } else if(dyna_narbs == kk+dyna_nummat2+dyna_nummat4+dyna_nummat8+dyna_nummatth ){
    nHeaderWords = 16;
  } else {
    throw("Inconsistent definition of dyna_narbs detected.");
  }
  */

  if(dyna_narbs == 0) return vector< vector<int> >();

  #ifdef QD_DEBUG
  cout << "Reading numbering ... ";
  #endif

  // pointer to nodes
  int nsort = buffer->read_int(wordPosition);
  // pointer to elem8 numbering
  int nsrh = buffer->read_int(wordPosition+1);
  if(nsrh != dyna_numnp+abs(nsort)) throw(string("nsrh != nsort + numnp is inconsistent in dyna file. Your file might be using FEMZIP."));
  // pointer to elem2 numbering
  int nsrb = buffer->read_int(wordPosition+2);
  if(nsrb != nsrh+dyna_nel8) throw(string("nsrb != nsrh + nel8 is inconsistent in dyna file."));
  // pointer to elem4 numbering
  int nsrs = buffer->read_int(wordPosition+3);
  if(nsrs != nsrb+dyna_nel2) throw(string("nsrs != nsrb + nel2 is inconsistent in dyna file."));
  // pointer to elemth numbering
  int nsrt = buffer->read_int(wordPosition+4);
  if(nsrt != nsrs+dyna_nel4) throw(string("nsrt != nsrs + nel4 is inconsistent in dyna file."));
  // nNode consistent?
  if(buffer->read_int(wordPosition+5) != dyna_numnp) throw(string("Number of nodes is not defined consistent in d3plot geometry section."));

  int nMaterials = dyna_nummat2+dyna_nummat4+dyna_nummat8+dyna_nummatth;

  /* === ID - ORDER === */
  // nodes,solids,beams,shells,tshells

  vector< vector<int> > idvector(4);

  // Node IDs
  if (nsort < 0) {
    wordPosition += 16;
  } else {
    wordPosition += 10;
  }
  //wordPosition += 16; // header length is 16
  wordsToRead = dyna_numnp;
  vector<int> nodeIDs(wordsToRead);
  size_t jj = 0;
  for(int ii=wordPosition;ii<wordPosition+wordsToRead;ii++){
    nodeIDs[jj] = buffer->read_int(ii);
    jj++;
  }
  idvector[0] = nodeIDs;

  // Solid IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nel8;
  vector<int> solidIDs(wordsToRead);
  jj = 0;
  for(int ii=wordPosition;ii<wordPosition+wordsToRead;ii++){
    solidIDs[jj] = buffer->read_int(ii);
    jj++;
  }
  idvector[1] = solidIDs;

  // Beam IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nel2;
  vector<int> beamIDs(wordsToRead);
  jj = 0;
  for(int ii=wordPosition;ii<wordPosition+wordsToRead;ii++){
    beamIDs[jj] = buffer->read_int(ii);
    jj++;
  }
  idvector[2] = beamIDs;

  // Shell IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nel4;
  vector<int> shellIDs(wordsToRead);
  jj = 0;
  for(int ii=wordPosition;ii<wordPosition+wordsToRead;ii++){
    shellIDs[jj] = buffer->read_int(ii);
    jj++;
  }
  idvector[3] = shellIDs;

  // Thick Shell IDs
  wordPosition += wordsToRead;
  wordPosition += dyna_nelth;
  // not read ...

	/*
   * Indeed this is a little complicated: usually the file should contain
   * as many materials as in the input but somehow dyna generates a few
   * ghost materials itself and those are appended with a 0 ID. Therefore
   * the length should be nMaterials but it's d3plot_nmmat with:
   * nMaterials < d3plot_nmmat. The difference are the ghost mats.
   * Took some time to find that out ...
   */

  vector<int> internalPartIDs(nMaterials);
  vector<int> externalPartIDs(nMaterials);
  wordsToRead = 3*dyna_nmmat;

  jj = 0;
  for(int ii=wordPosition+dyna_nmmat;ii<wordPosition+dyna_nmmat+nMaterials;ii++){
    externalPartIDs[jj] = buffer->read_int(ii);
  }

  jj = 0;
  for(int ii=wordPosition+2*dyna_nmmat;ii<wordPosition+2*dyna_nmmat+nMaterials;ii++){
    internalPartIDs[jj] = buffer->read_int(ii);
  }

  // update position
  wordPosition += wordsToRead;
  /*
   * the whole numbering section should have length narbs
   * but the procedure here should do the same ... hopefully
   */
  //wordPosition += dyna_narbs;

  // extra node elements
  // 10 node solids: 2 node conn
  if(own_nel10)
    wordPosition += 2*abs(dyna_nel8);
  // 8 node shells: 4 node conn
  if(dyna_nel48 > 0)
    wordPosition += 5*dyna_nel48;
  // 20 node solids: 12 node conn
  if((dyna_extra > 0) && (dyna_nel20 > 0))
    wordPosition += 13*dyna_nel20;
  #ifdef QD_DEBUG
  cout << "done." << endl;
  #endif

  return std::move(idvector);

}

/*
 * Read the part names from the geometry section.
 */
void D3plot::read_geometry_parts(){

  #ifdef QD_DEBUG
  cout << "Reading parts ... " ;
  #endif

  int ntype = this->buffer->read_int(wordPosition);
  if (ntype != 90001){
    throw(string("ntype must be 90001 in part section."));
  }

  this->dyna_numprop = this->buffer->read_int(wordPosition+1);
  if (this->dyna_numprop < 0)
    throw(string("negative number of parts in part section makes no sense."));
  for(int ii=0; ii<this->dyna_numprop; ii++){

    int start = (wordPosition + 1) + ii*19 + 1;
    int partID = this->buffer->read_int(start);
    string partName = this->buffer->read_str(start+1,18);

    this->get_db_parts()->add_part_byID(partID)->set_name(partName);

  }

  #ifdef QD_DEBUG
  cout << this->get_db_parts()->size() << " done." << endl;
  #endif

  // update position
  wordPosition += 1 + (this->dyna_numprop+1)*19+1;

}


/*
 * Check for the file ending marker. That is a float
 * of -999999.
 */
bool D3plot::isFileEnding(int iWord){
  if (this->buffer->read_float(iWord)+999999.0f == 0.)
    return true;
  return false;
}


/*
 * Parse the user input for state reading.
 *
 */
void D3plot::read_states_parse(vector<string> _variables){

  // Safety reset
  this->disp_read = 0;
  this->vel_read = 0;
  this->acc_read = 0;
  this->stress_read = 0;
  this->stress_mises_read = 0;
  this->strain_read = 0;
  this->energy_read = 0;
  this->plastic_strain_read = 0;

  this->history_shell_read.clear();
  this->history_shell_mode.clear();

  this->history_solid_read.clear();
  this->history_solid_mode.clear();

  for(size_t ii=0; ii<_variables.size(); ++ii){
    // Displacement
    if(_variables[ii].find("disp") != string::npos){
      if( dyna_iu == 0)
        throw(string("Unable to read displacements, since there are none."));
      this->disp_read = read_states_parse_readMode(_variables[ii]);

      if(this->disp_is_read){
        #ifdef QD_DEBUG
        cout << "disp already loaded." << endl;
        #endif
        this->disp_read = 0;
      }
    // Velocity
    } else if(_variables[ii].find("vel") != string::npos){
      if( dyna_iv == 0)
        throw(string("Unable to read velocities, since there are none."));
      this->vel_read = read_states_parse_readMode(_variables[ii]);

      if(this->vel_is_read){
        #ifdef QD_DEBUG
        cout << "vel already loaded." << endl;
        #endif
        this->vel_read = 0;
      }
    // Acceleration
    } else if(_variables[ii].find("accel") != string::npos){
      if( dyna_ia == 0)
        throw(string("Unable to read accelerations, since there are none."));
      this->acc_read = read_states_parse_readMode(_variables[ii]);

      if(this->acc_is_read){
        #ifdef QD_DEBUG
        cout << "accel already loaded." << endl;
        #endif
        this->acc_read = 0;
      }
    // Mises Stress ( must be before stress! )
    } else if(_variables[ii].find("stress_mises") != string::npos){
      #ifdef QD_DEBUG
      if( dyna_ioshl1 == 0)
        cout << "Warning: There are no shell-stresses in the file." << endl;
      #endif
      this->stress_mises_read = read_states_parse_readMode(_variables[ii]);

      if(this->stress_mises_is_read){
        #ifdef QD_DEBUG
        cout << "stress_mises already loaded." << endl;
        #endif
        this->stress_mises_read = 0;
      }

    // Stress
    } else if(_variables[ii].find("stress") != string::npos){
      #ifdef QD_DEBUG
      if( dyna_ioshl1 == 0)
        cout << "Warning: There are no shell-stresses in the file." << endl;
      #endif
      this->stress_read = read_states_parse_readMode(_variables[ii]);

      if(this->stress_is_read){
        #ifdef QD_DEBUG
        cout << "stress already loaded." << endl;
        #endif
        this->stress_read = 0;
      }
    // Plastic Strain
    // must be before strain !!!!
    } else if(_variables[ii].find("plastic_strain") != string::npos){
      #ifdef QD_DEBUG
      if( dyna_ioshl2 == 0)
        cout << "Warning: There are no shell-plastic-strains in the file." << endl;
      #endif
      this->plastic_strain_read = read_states_parse_readMode(_variables[ii]);

      if(this->plastic_strain_is_read){
        #ifdef QD_DEBUG
        cout << "plastic strain already loaded." << endl;
        #endif
        this->plastic_strain_read = 0;
      }
    // Strain
    } else if(_variables[ii].find("strain") != string::npos){
      if( dyna_istrn == 0)
        throw(string("Unable to read strains, since there are none."));
      this->strain_read = read_states_parse_readMode(_variables[ii]);

      if(this->strain_is_read){
        #ifdef QD_DEBUG
        cout << "strain already loaded." << endl;
        #endif
        this->strain_read = 0;
      }
    // Internal Energy
    } else if(_variables[ii].find("energy") != string::npos){
      if( dyna_ioshl4 == 0)
        throw(string("Unable to read energies, since there are none."));
      this->energy_read = read_states_parse_readMode(_variables[ii]);

      if(this->energy_is_read){
        #ifdef QD_DEBUG
        cout << "energy already loaded." << endl;
        #endif
        this->energy_read = 0;
      }
    // History variables
    } else if(_variables[ii].find("history") != string::npos){

      // retrieve history var indexes
      vector<unsigned int> hist_vars = extract_integers<unsigned int>(_variables[ii]);
      if(hist_vars.size() < 1)
         throw(string("No history variable index specified. Please input at least one number seperated by spaces."));
      unsigned int var_mode = read_states_parse_readMode(_variables[ii]);

      /* SHELLS */
      if(_variables[ii].find("shell") != string::npos){

         // Check: already loaded
         for(size_t jj=0; jj<this->history_shell_is_read.size(); ++jj){
            vector<unsigned int>::iterator kk = find(hist_vars.begin(), hist_vars.end(), this->history_shell_is_read[jj]);
            if ( kk != hist_vars.end() ){
               #ifdef QD_DEBUG
               cout << "history variable " << *kk << " already loaded for shells." << endl;
               #endif
               hist_vars.erase(kk);
            }
         }

         // Check: already defined in an argument previously
         for(size_t jj=0; jj<this->history_shell_read.size(); ++jj){
            vector<unsigned int>::iterator kk = find(hist_vars.begin(), hist_vars.end(), this->history_shell_read[jj]);
            if ( kk != hist_vars.end() ){
               cout << "Warning: trying to read history variable " << *kk << " twice for shells, using only first occurrence." << endl;
               hist_vars.erase(kk);
            }
         }

         // Check: var index beyond limit neips
         for(size_t jj=0; jj < hist_vars.size(); ++jj){

            if( hist_vars[jj] < 1){
               throw(string("History variable index must be at least 1."));
            }
            if( static_cast<int>(hist_vars[jj]) > this->dyna_neips ){
               cout << "Warning: history variable " << hist_vars[jj] << " exceeds the limit for shells of " << this->dyna_neips << endl;
            }
         }

         // Save var indexes and var modes
         for(size_t jj=0; jj<hist_vars.size(); ++jj){
            this->history_shell_read.push_back(hist_vars[jj]);
            this->history_shell_mode.push_back(var_mode);
         }

      /* SOLIDS */
      } else if(_variables[ii].find("solid") != string::npos) {

         // Check: already loaded
         for(size_t jj=0; jj<this->history_solid_is_read.size(); ++jj){
            vector<unsigned int>::iterator kk = find(hist_vars.begin(), hist_vars.end(), this->history_solid_is_read[jj]);
            if ( kk != hist_vars.end() ){
               #ifdef QD_DEBUG
               cout << "history variable " << *kk << " already loaded for solids." << endl;
               #endif
               hist_vars.erase(kk);
            }
         }

         // Check: already defined in an argument previously
         for(size_t jj=0; jj<this->history_solid_read.size(); ++jj){
            vector<unsigned int>::iterator kk = find(hist_vars.begin(), hist_vars.end(), this->history_solid_read[jj]);
            if ( kk != hist_vars.end() ){
               cout << "Warning: trying to read history variable " << *kk << " twice for solids, using only first occurrence." << endl;
               hist_vars.erase(kk);
            }
         }

         // Check: var index beyond limit neiph
         for(size_t jj=0; jj < hist_vars.size(); ++jj){

            if(hist_vars[jj] < 1){
               throw(string("History variable index must be at least 1."));
            }
            if( static_cast<int>(hist_vars[jj]) > this->dyna_neiph ){
               cout << "Warning: history variable " << hist_vars[jj] << " exceeds the limit for solids of " << this->dyna_neiph << endl;
            }
         }

         // save var indexes
         for(size_t jj=0; jj<hist_vars.size(); ++jj){
            this->history_solid_read.push_back(hist_vars[jj]);
            this->history_solid_mode.push_back(var_mode);
         }

      // unknown element type
      } else {
         throw(string("Please specify the element type for all history variables as shell or solid"));
      }

    } else {
        throw(string("Reading of variable:")+_variables[ii]+string(" is undefined"));
    } // if:variable.find
  } //for:variables

}

  /*
   * Returns the int code for the read mode of the state variables in the d3plot.
   * Modes are: min,max,outer,mid,inner,mean
   */
unsigned int D3plot::read_states_parse_readMode(const string& _variable) const {

   if(_variable.find("max") != string::npos){
     return 1;
   } else if(_variable.find("min") != string::npos){
     return 2;
   } else if(_variable.find("outer") != string::npos){
     return 3;
   } else if(_variable.find("mid") != string::npos){
     return 4;
   } else if(_variable.find("inner") != string::npos){
     return 5;
   } else if(_variable.find("mean") != string::npos){
     return 6;
   } else {
     return 6; // std is mean
   }

}


/** Read the state data.
 * 
 * @param _variables vector of variables to read
 */
void D3plot::read_states(vector<string> _variables){

  #ifdef QD_DEBUG
  cout << "> STATES" << endl;
  for(size_t ii=0; ii<_variables.size(); ++ii)
    cout << "variable: " << _variables[ii] << endl;
  #endif

  if( (_variables.size() < 1) && (this->timesteps.size() > 0))
    throw(string("The list of state variables to load is empty."));

  // Decode variable reading
  this->read_states_parse(_variables);

  // Check if variables are already read.
  // If just 1 is not read yet, the routine runs through.
  // Will not work first time :P since we must run through at least once
  if( (this->disp_read
     +this->vel_read
     +this->acc_read
     +this->plastic_strain_read
     +this->energy_read
     +this->strain_read
     +this->stress_read
     +this->stress_mises_read
     +this->history_shell_read.size()
     +this->history_solid_read.size() == 0) && (this->timesteps.size() != 0))
      return;

  // Calculate loop properties
  size_t iState = 0;
  int nVarsNodes = dyna_ndim*(dyna_iu+dyna_iv+dyna_ia)*dyna_numnp;
  int nVarsElems = dyna_nel2*dyna_nv1d+(dyna_nel4-dyna_numrbe)*dyna_nv2d+dyna_nel8*dyna_nv3d;

  // Variable Deletion table
  int nDeletionVars = 0;
  if (dyna_mdlopt == 0){
    // ok
  } else if(dyna_mdlopt == 1){
    nDeletionVars = dyna_numnp;
  } else if(dyna_mdlopt == 2){
    nDeletionVars = dyna_nel2+dyna_nel4+dyna_nel8+dyna_nelth;
  } else {
    throw("Parameter mdlopt:"+to_string(dyna_mdlopt)+" makes no sense.");
  }

  // Checks for timesteps
  bool timesteps_read = false;
  if(this->timesteps.size() < 1)
    timesteps_read = true;

  bool firstFileDone = false;

  // Check for first time
  // Makes no difference for D3plotBuffer but for
  // the FemzipBuffer.
  if(this->timesteps.size() < 1){
    this->buffer->init_nextState();
    this->wordPositionStates = this->wordPosition;
  } else{
    this->buffer->rewind_nextState();
    this->wordPosition = this->wordPositionStates;
  }

   // Loop over state files
  while(this->buffer->has_nextState()){

    this->buffer->read_nextState();

    // Not femzip case
    if((!this->useFemzip) && firstFileDone){
      wordPosition = 0;
    }
    // femzip case
    if(this->useFemzip){
      // 0 = endmark
      // 1 = ntype = 90001
      // 2 = numprop
      int dyna_numprop_states = this->buffer->read_int(2);
      if(this->dyna_numprop != dyna_numprop_states)
        throw(string("Numprop in geometry section != numprop in states section!"));
      wordPosition = 1; // endline symbol at 0 in case of femzip ...
      wordPosition += 1 + (this->dyna_numprop+1)*19 + 1;
      //this->femzip_state_offset = wordPosition;
    }

    // Loop through states
    while(!this->isFileEnding(wordPosition)){

      if(timesteps_read){
        float state_time = buffer->read_float(wordPosition);
        this->timesteps.push_back(state_time);
        #ifdef QD_DEBUG
        cout << "State: " << iState << " Time: " << state_time << endl;
        #endif
      }

      // NODE - DISP
      if(dyna_iu && (this->disp_read != 0) ){
        read_states_displacement();
      }

      // NODE - VEL
      if(dyna_iv && (this->vel_read != 0) ){
        read_states_velocity();
      }

      // NODE - ACCEL
      if(dyna_ia && (this->acc_read != 0) ){
        read_states_acceleration();
      }

      // ELEMENT - STRESS, STRAIN, ENERGY, PLASTIC STRAIN
      if( this->stress_read
         || this->stress_mises_read
         || this->strain_read 
         || this->energy_read
         || this->plastic_strain_read
         || this->history_shell_read.size()
         || this->history_solid_read.size()){

        // Element 4
        read_states_elem4(iState);
        // Element 8
        read_states_elem8(iState);

      }

      // update position
      wordPosition += nVarsNodes + nVarsElems + nDeletionVars + dyna_nglbv + 1;

      iState++;
    }

    firstFileDone = true;
  }

  this->buffer->end_nextState();

  // Set, which variables were read
  if( this->disp_read != 0 ){
    this->disp_is_read = true;
  }
  if( this->vel_read != 0 ){
    this->vel_is_read = true;
  }
  if( this->plastic_strain_read != 0 ){
    this->plastic_strain_is_read = true;
  }
  if( this->energy_read != 0 ){
    this->energy_is_read = true;
  }
  if( this->strain_read != 0 ){
    this->strain_is_read = true;
  }
  if( this->stress_read != 0 ){
    this->stress_is_read = true;
  }
  if( this->stress_mises_read != 0 ){
    this->stress_mises_is_read = true;
  }
  for(size_t ii=0; ii<this->history_shell_read.size(); ++ii){
     this->history_shell_is_read.push_back(this->history_shell_read[ii]);
  }
  for(size_t ii=0; ii<this->history_solid_read.size(); ++ii){
     this->history_solid_is_read.push_back(this->history_solid_read[ii]);
  }

}


/*
 * Read the node displacement into the db.
 *
 */
void D3plot::read_states_displacement(){

  if(dyna_iu != 1)
    return;

  int start = wordPosition + dyna_nglbv + 1;
  wordsToRead = dyna_numnp*dyna_ndim;
  size_t iNode = 0;

  DB_Nodes *db_nodes = this->get_db_nodes();
  vector<float> _disp(dyna_ndim);

  for(int ii = start;ii < start+wordsToRead; ii+=dyna_ndim){

    Node* node = db_nodes->get_nodeByIndex(iNode);
    
    //_disp.clear();
    buffer->read_float_array(ii, dyna_ndim, _disp);
    /*for(int jj=ii; jj<ii+dyna_ndim; ++jj)
      _disp.push_back(this->buffer->read_float(jj));
      */

    node->add_disp(_disp);

    ++iNode;
  }

}


/*
 * Read the node velocity.
 *
 */
void D3plot::read_states_velocity(){

  if(dyna_iv != 1)
    return;

  int start = 1 + dyna_nglbv + (dyna_iu) * dyna_numnp * dyna_ndim + wordPosition;
  wordsToRead = dyna_numnp*dyna_ndim;
  size_t iNode = 0;

  DB_Nodes *db_nodes = this->get_db_nodes();
  vector<float> _vel(dyna_ndim);

  for(int ii = start;ii < start+wordsToRead; ii+=dyna_ndim){

    Node* node = db_nodes->get_nodeByIndex(iNode);
    
    //_vel.clear();
    buffer->read_float_array(ii, dyna_ndim, _vel);
    /*
    for(int jj=ii; jj<ii+dyna_ndim; ++jj)
      _vel.push_back(this->buffer->read_float(jj));
    */

    node->add_vel(_vel);

    ++iNode;
  }

}


/*
 * Read the node acceleration.
 *
 */
void D3plot::read_states_acceleration(){

  if(dyna_ia != 1)
    return;

  int start = 1 + dyna_nglbv + (dyna_iu+dyna_iv) * dyna_numnp * dyna_ndim + wordPosition;
  wordsToRead = dyna_numnp*dyna_ndim;
  int iNode = 0;

  DB_Nodes *db_nodes = this->get_db_nodes();
  vector<float> _accel(dyna_ndim);

  for(int ii = start;ii < start+wordsToRead; ii+=dyna_ndim){

    Node* node = db_nodes->get_nodeByIndex(iNode);
    
    //_accel.clear();
    buffer->read_float_array(ii, dyna_ndim, _accel);
    /*
    for(int jj=ii; jj<ii+dyna_ndim; ++jj)
      _accel.push_back(this->buffer->read_float(jj));
    */

    node->add_accel(_accel);

    ++iNode;
  }

}


/*
 * Read the data of the 8 node solids.
 * > Strain Tensor
 * > Strain Mises
 * > Stress Tensor
 * > Stress Mises
 * > Eq. Plastic Strain 
 *
 */
void D3plot::read_states_elem8(size_t iState){

  if((dyna_nv3d <= 0) && (dyna_nel8 <= 0))
    return;

  int start = this->wordPosition 
             + 1 // time
             + dyna_nglbv 
             + (dyna_iu+dyna_iv+dyna_ia) * dyna_numnp * dyna_ndim;
  wordsToRead = dyna_nv3d * dyna_nel8;

  vector<float> tmp_vector(6);
  vector<float> history_vars(this->history_solid_read.size());
  DB_Elements *db_elements = this->get_db_elements();

  size_t iElement = 0;
  for(int ii = start;ii < start+wordsToRead; ii+=dyna_nv3d){

    Element* element = db_elements->get_elementByIndex(SOLID,iElement);

    // stress tensor and data
    if(this->stress_read || this->stress_mises_read){
      //tmp_vector.clear();
      buffer->read_float_array(ii, 6, tmp_vector);
      /*
      tmp_vector.push_back(this->buffer->read_float(ii));
      tmp_vector.push_back(this->buffer->read_float(ii+1));
      tmp_vector.push_back(this->buffer->read_float(ii+2));
      tmp_vector.push_back(this->buffer->read_float(ii+3));
      tmp_vector.push_back(this->buffer->read_float(ii+4));
      tmp_vector.push_back(this->buffer->read_float(ii+5));
      */

      if(this->stress_read)
        element->add_stress(tmp_vector);
      if(this->stress_mises_read)
        element->add_stress_mises(MathUtility::mises_stress(tmp_vector));
    }

    // plastic strain
    if(this->plastic_strain_read){
      element->add_plastic_strain(this->buffer->read_float(ii+6));
    }

    // strain tensor
    if( (dyna_istrn == 1) && this->strain_read ){
      //tmp_vector.clear();
      buffer->read_float_array(ii+dyna_nv3d-6, 6, tmp_vector);
      /*
      tmp_vector.push_back(this->buffer->read_float(ii+dyna_nv3d-6));
      tmp_vector.push_back(this->buffer->read_float(ii+dyna_nv3d-5));
      tmp_vector.push_back(this->buffer->read_float(ii+dyna_nv3d-4));
      tmp_vector.push_back(this->buffer->read_float(ii+dyna_nv3d-3));
      tmp_vector.push_back(this->buffer->read_float(ii+dyna_nv3d-2));
      tmp_vector.push_back(this->buffer->read_float(ii+dyna_nv3d-1));
      */
      element->add_strain(tmp_vector);
    }

    // no energy ...

    // history variables
    if(history_solid_read.size()){
      history_vars.clear();

      for(size_t jj=0; jj<history_solid_read.size(); ++jj){

        // Skip if over limit
        if(this->history_solid_read[jj] > this->dyna_neiph)
           continue;

        history_vars.push_back(this->buffer->read_float(ii+6+history_solid_read[jj]));

      } // loop:history
      element->add_history_vars(history_vars, iState);
    } // if:history

    iElement++;
  }

}


/*
 * Read the data of the 4 node solids.
 * > Strain Tensor
 * > Strain Mises
 * > Stress Tensor
 * > Stress Mises
 * > Eq. Plastic Strain (done)
 *
 */
void D3plot::read_states_elem4(size_t iState){

  if((dyna_istrn != 1) && (dyna_nv2d <= 0) && (dyna_nel4-dyna_numrbe>0))
    return;

  // prepare looping
  int start = this->wordPosition 
             + 1 // time 
             + dyna_nglbv
             + (dyna_iu+dyna_iv+dyna_ia) * dyna_numnp * dyna_ndim
             + dyna_nv3d * dyna_nel8
             + dyna_nv1d * dyna_nel2;
  wordsToRead = dyna_nv2d*(dyna_nel4-dyna_numrbe);

  // offsets
  int iPlastStrainOffset = this->dyna_ioshl1*6; // stresses before?
  int iHistoryOffset     = iPlastStrainOffset + this->dyna_ioshl2; // stresses & pl. strain before
  int iLayerSize         = dyna_neips + iHistoryOffset;

  // helpful vars
  float dyna_maxint_float = (float) dyna_maxint;
  float stress_mises = 0.f;
  float plastic_strain = 0.f;
  vector<float> tmp_vec_stress(6);
  vector<float> tmp_vec6(6);
  vector<float> history_vars(this->history_shell_read.size());

  // Do the thing ... 
  size_t iElement = 0;
  for(int ii = start; ii < start+wordsToRead; ++iElement){

    // get element (and check for rigidity)
    Element *element = this->get_db_elements()->get_elementByIndex(SHELL, iElement);
    if( element->get_is_rigid() ){
      // does not increment ii, but iElement!!!!!
      continue;
    }

    // preallocate layer vars
    stress_mises = 0.f;
    plastic_strain = 0.f;
    if(this->history_shell_read.size()){
      history_vars.assign(this->history_shell_read.size(),0.f);
    }

    // LOOP: LAYERS
    for(int iLayer = 0; iLayer < dyna_maxint; ++iLayer){

      int layerStart = ii + iLayer*iLayerSize;

      // LAYER: PLASTIC_STRAIN
      if( (this->plastic_strain_read) && (dyna_ioshl2)){
        float _tmp = this->buffer->read_float(layerStart+iPlastStrainOffset);

        if(iLayer == 0){
          plastic_strain = _tmp;

        } else {
          
          switch(this->plastic_strain_read){
            case 1: //max
              plastic_strain = (plastic_strain>_tmp) ? plastic_strain : _tmp;
              break;
            case 2: //min
              plastic_strain = (plastic_strain<_tmp) ? plastic_strain : _tmp;
              break;
            case 3: //out
              if(iLayer == dyna_maxint -1)
                plastic_strain = _tmp;
              break;
            case 4: //mid
              if(iLayer == (int) (dyna_maxint_float / 2.f))
                plastic_strain = _tmp;
              break;
            case 5: //in
              // already ok
              break;
            case 6: //mean
              plastic_strain += _tmp;
              if( iLayer == dyna_maxint -1 )
                plastic_strain /= dyna_maxint_float;
              break;
            default:
              break;
          }// end:switch
        } // end:iLayer!=0
      }

      // LAYER: STRESS TENSOR AND MISES
      if( (this->stress_read || this->stress_mises_read) && (dyna_ioshl1) ){

        //tmp_vec6.clear();
        buffer->read_float_array(layerStart, 6, tmp_vec6);
        /*
        tmp_vec6.push_back(this->buffer->read_float(layerStart));
        tmp_vec6.push_back(this->buffer->read_float(layerStart+1));
        tmp_vec6.push_back(this->buffer->read_float(layerStart+2));
        tmp_vec6.push_back(this->buffer->read_float(layerStart+3));
        tmp_vec6.push_back(this->buffer->read_float(layerStart+4));
        tmp_vec6.push_back(this->buffer->read_float(layerStart+5));
        */

        if(iLayer == 0){
          tmp_vec_stress = tmp_vec6;
          stress_mises = MathUtility::mises_stress(tmp_vec6);
        } else {

          // stress tensor
          switch( this->stress_read ){
            case 1 : // max
              for(int jj=0; jj<6; jj++)
                tmp_vec_stress[jj] = tmp_vec6[jj] > tmp_vec_stress[jj] 
                                     ? tmp_vec6[jj] : tmp_vec_stress[jj];
              break;
            case 2: // min
              for(int jj=0; jj<6; jj++)
                tmp_vec_stress[jj] = tmp_vec6[jj] < tmp_vec_stress[jj] 
                                     ? tmp_vec6[jj] : tmp_vec_stress[jj];
              break;
            case 3: // outer
              if(iLayer == dyna_maxint - 1)
                tmp_vec_stress = tmp_vec6;
              break;
            case 4: // mid
              if(iLayer == (int) ( dyna_maxint_float / 2.f) )   
                tmp_vec_stress = tmp_vec6;
              break;
            case 5: // inner
              // nothing
              break;
            case 6: // mean
              for(int jj=0; jj<6; jj++)
                tmp_vec_stress[jj] += tmp_vec6[jj];
              
              if(iLayer == dyna_maxint-1)
                for(int jj=0; jj<6; jj++)
                  tmp_vec_stress[jj] /= dyna_maxint_float;
              
              break;
            default: // none
              break;
          } // end:switch(stress_read)

          // stress mises calculation
          switch( this->stress_mises_read ){
            case 1 : {// max
              float _tmp2 = MathUtility::mises_stress(tmp_vec6);
              stress_mises = _tmp2 > stress_mises ? _tmp2 : stress_mises;
              break;
              }
            case 2: {// min
              float _tmp2 = MathUtility::mises_stress(tmp_vec6);
              stress_mises = _tmp2 < stress_mises ? _tmp2 : stress_mises;
              break;
              }
            case 3: // outer
              if(iLayer == dyna_maxint - 1)
                stress_mises =  MathUtility::mises_stress(tmp_vec6);
              break;
            case 4: // mid
              if(iLayer == (int) ( dyna_maxint_float / 2.f) )
                stress_mises =  MathUtility::mises_stress(tmp_vec6);
              break;
            case 5: // inner
              // nothing
              break;
            case 6: // mean
              stress_mises += MathUtility::mises_stress(tmp_vec6);
              if(iLayer == dyna_maxint-1)
                stress_mises /= dyna_maxint_float;
              break;
            default: // none
              break;
          } // end:switch(stress_mises_read)
        } // end:iLayer!=0
      } // end:stress

      // LAYERS: HISTORY SHELL
      for(size_t jj=0; jj<this->history_shell_read.size(); ++jj){

         // Skip if over limit
         if(this->history_shell_read[jj] > this->dyna_neips)
            continue;

        // max
        if(this->history_shell_mode[jj] == 1) {
          float _tmp;
          _tmp = this->buffer->read_float(layerStart+6+history_shell_read[jj]);
          if(iLayer == 0){
            history_vars[jj] = _tmp;
          } else {
            history_vars[jj] = (_tmp > history_vars[jj]) ? _tmp : history_vars[jj];
          }
        // min
        } else if(this->history_shell_mode[jj] == 2) {
          float _tmp;
          _tmp = this->buffer->read_float(layerStart+6+history_shell_read[jj]);
          if(iLayer == 0){
            history_vars[jj] = _tmp;
          }else {
            history_vars[jj] = (_tmp < history_vars[jj]) ? _tmp : history_vars[jj];
          }
        // out
        } else if((this->history_shell_mode[jj] == 3) && (iLayer == dyna_maxint - 1) ) {
          history_vars[jj] = this->buffer->read_float(layerStart+6+history_shell_read[jj]);
        // mid
        } else if((this->history_shell_mode[jj] == 4) && (iLayer == (int) (dyna_maxint_float / 2.f)) ) {
          history_vars[jj] = this->buffer->read_float(layerStart+6+history_shell_read[jj]);
        // in
        } else if((this->history_shell_mode[jj] == 5) && (iLayer == 1) ) {
          history_vars[jj] = this->buffer->read_float(layerStart+6+history_shell_read[jj]);
        // mean
        } else if((this->history_shell_mode[jj] == 6)) {

          history_vars[jj] += this->buffer->read_float(layerStart+6+history_shell_read[jj]);
          if(iLayer == dyna_maxint-1){
             float _tmp = (float) dyna_maxint;
             history_vars[jj] /= _tmp;
          }

        }
      } // loop:history
    } // loop:layers

    // add layer vars (if requested)
    if(dyna_istrn && this->plastic_strain_read)
      element->add_plastic_strain(plastic_strain);
    if( this->stress_read )
      element->add_stress(tmp_vec_stress);
    if( this->stress_mises_read )
      element->add_stress_mises(stress_mises);
    if(this->history_shell_read.size())
      element->add_history_vars(history_vars,iState);

    // STRAIN TENSOR
    if((dyna_istrn == 1) && this->strain_read ){

      vector<float> strain(6);
      int strainStart = (dyna_nv2d >= 45) ? ii+dyna_nv2d - 13 : ii+dyna_nv2d - 12;

      // max
      if(this->strain_read == 1) {

        float _tmp;
        float _tmp2;
        for(int jj=0; jj<6; jj++){
          _tmp  = this->buffer->read_float(strainStart+jj);
          _tmp2 = this->buffer->read_float(strainStart+jj+6);
          strain[jj] = (_tmp > _tmp2) ? _tmp : _tmp2;
        }

      // min
      } else if(this->strain_read == 2) {

        float _tmp;
        float _tmp2;
        for(int jj=0; jj<6; jj++){
          _tmp  = this->buffer->read_float(strainStart+jj);
          _tmp2 = this->buffer->read_float(strainStart+jj+6);
          strain[jj] = (_tmp < _tmp2) ? _tmp : _tmp2;
        }

      // out
      } else if(this->strain_read == 3) {

        for(int jj=0; jj<6; jj++){
          strain[jj] = this->buffer->read_float(strainStart+6+jj);
        }

      // in
      } else if(this->strain_read == 5) {

        for(int jj=0; jj<6; jj++){
          strain[jj] = this->buffer->read_float(strainStart+jj);
        }

      // mean/mid
      } else if( (this->strain_read == 6) | (this->strain_read == 4) ) {

        for(int jj=0; jj<6; jj++){
          strain[jj] = ( this->buffer->read_float(strainStart+jj)
                        +this->buffer->read_float(strainStart+jj+6))/2.f;
        }

      }

      element->add_strain(strain);
    }

    // internal energy (a little complicated ... but that's dyna)
    if( this->energy_read && (dyna_ioshl4)) {
      if((dyna_istrn == 1)){
        if(dyna_nv2d >=45)
          element->add_energy(this->buffer->read_float(ii+dyna_nv2d-1));
      } else {
        element->add_energy(this->buffer->read_float(ii+dyna_nv2d-1));
      }
    }

    ii+=dyna_nv2d;
  }

}


/** Get the timestamps of the timesteps.
 *
 * @return timesteps vector with the timestamp of the given state
 */
vector<float> D3plot::get_timesteps() {
  return this->timesteps;
}



/** Tells whether displacements were loaded.
 * 
 * @return disp_is_read boolean whether the disp was read
 */
bool D3plot::displacement_is_read(){
   return this->disp_is_read;
}


/** Clears loaded result data loaded from the file
 *
 * @param _variables optional arg for cleansing only specific variables
 *
 * Clears all variables by default!
 */
void D3plot::clear( const vector<string>& _variables ){

  // Default: Clear all
  if( _variables.size() == 0 ) {

    // hihi this is a naughty trick ... calling myself again
    vector<string> _tmp;
    _tmp.push_back("disp");
    _tmp.push_back("vel");
    _tmp.push_back("accel");
    _tmp.push_back("energy");
    _tmp.push_back("plastic_strain");
    _tmp.push_back("strain");
    _tmp.push_back("stress");
    _tmp.push_back("stress_mises");
    _tmp.push_back("history shell");
    _tmp.push_back("history solid");
    this->clear( _tmp );

  } else {

    // Convert strings to booleans (faster later)
    bool delete_disp  = false;
    bool delete_vel   = false;
    bool delete_accel = false;
    bool delete_energy = false;
    bool delete_plastic_strain = false;
    bool delete_strain = false;
    bool delete_stress = false;
    bool delete_stress_mises = false;
    bool delete_history_shell = false;
    bool delete_history_solid = false;
    for(size_t iVar=0; iVar<_variables.size(); ++iVar){

      if( _variables[iVar].find("disp") != string::npos ){
        delete_disp = true;
      } else if ( _variables[iVar].find("vel") != string::npos ) {
        delete_vel = true;
      } else if ( _variables[iVar].find("accel") != string::npos ) {
        delete_accel = true;
      } else if ( _variables[iVar].find("energy") != string::npos ) {
        delete_energy = true;
      } else if ( _variables[iVar].find("plastic_strain") != string::npos ) {
        delete_plastic_strain = true;
      } else if ( _variables[iVar].find("strain") != string::npos ) {
        delete_strain = true;
      } else if ( _variables[iVar].find("stress_mises") != string::npos ) {
        delete_stress_mises = true;
      } else if ( _variables[iVar].find("stress") != string::npos ) {
        delete_stress = true;
      } else if ( _variables[iVar].find("history") != string::npos ) {

        // shell or solid specified?
        // delete both if unspecified
        if( _variables[iVar].find("shell") != string::npos ){
          delete_history_shell = true;
        } else if ( _variables[iVar].find("solid") != string::npos ){
          delete_history_solid = true;
        } else {
          delete_history_shell = true;
          delete_history_solid = true;
        }

      } else {
        throw(string("Unknown variable type:")+_variables[iVar]);
      }

    } // end:for

    // NODES: data deletion
    if( delete_disp || delete_vel || delete_accel ){

      DB_Nodes *db_nodes = this->get_db_nodes();
      Node *_node = nullptr;
      for(size_t iNode=0; iNode<db_nodes->size(); ++iNode){

        _node = db_nodes->get_nodeByIndex(iNode);
        if( _node ){
          
          if(delete_disp)
            _node->clear_disp();
          if(delete_vel)
            _node->clear_vel();
          if(delete_accel)
            _node->clear_accel();

        }

      } // end:for

      // reset flags
      if(delete_disp)
        this->disp_is_read = false;
      if(delete_vel)
        this->vel_is_read = false;
      if(delete_accel)
        this->acc_is_read = false;
    }

    // ELEMENT: data deletion
    if( delete_energy ||
        delete_plastic_strain ||
        delete_strain ||
        delete_stress ||
        delete_stress_mises ||
        delete_history_shell ||
        delete_history_solid ){

      DB_Elements *db_elems = this->get_db_elements();
      Element *_elem = nullptr;

      // shells      
      for(size_t iElement=0; iElement<db_elems->size(SHELL); iElement++){

        _elem = db_elems->get_elementByIndex(SHELL, iElement);
        if( _elem ){
          if(delete_energy)
            _elem->clear_energy();
          if(delete_plastic_strain)
            _elem->clear_plastic_strain();
          if(delete_strain)
            _elem->clear_strain();
          if(delete_stress)
            _elem->clear_stress();
          if(delete_stress_mises)
            _elem->clear_stress_mises();
          if(delete_history_shell)
            _elem->clear_history_vars();
        }
      }
      // solids
      for(size_t iElement=0; iElement<db_elems->size(SOLID); iElement++){
      
        _elem = db_elems->get_elementByIndex(SOLID, iElement);
        if( _elem ){
          if(delete_energy)
            _elem->clear_energy();
          if(delete_plastic_strain)
            _elem->clear_plastic_strain();
          if(delete_strain)
            _elem->clear_strain();
          if(delete_stress)
            _elem->clear_stress();
          if(delete_stress_mises)
            _elem->clear_stress_mises();
          if(delete_history_solid)
            _elem->clear_history_vars();
        }
      }

      // reset flags
      if(delete_energy)
        this->energy_is_read = false;
      if(delete_plastic_strain)
        this->plastic_strain_is_read = false;
      if(delete_strain)
        this->strain_is_read = false;
      if(delete_stress)
        this->stress_is_read = false;
      if(delete_stress_mises)
        this->stress_mises_is_read = false;
      if(delete_history_shell)
        this->history_shell_is_read.clear();
      if(delete_history_solid)
        this->history_solid_is_read.clear();

    } // end:if Elements
  } // end:else for deletion
} // end:function clear
