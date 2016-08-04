
#ifndef D3PLOT
#define D3PLOT

// forward declarations
class DB_Nodes;
class DB_Parts;
class DB_Elements;

#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <algorithm> // trim

#include "AbstractBuffer.h"

using namespace std;

class D3plot {

  private:

  string filename;
  string dyna_title;
  string dyna_datetime; // BUGGY

  int dyna_ndim; // dimension parameter
  int dyna_numnp; // number of nodes
  int dyna_mdlopt; // describes element deletion
  int dyna_mattyp; // material types section is read

  int dyna_nglbv; // global vars per timestep

  int dyna_nel2; // #elements with 2 nodes (beams)
  int dyna_nel4; // #elements with 4 nodes (shells)
  int dyna_nel48; // # 8 node shell elements?!?!?!
  int dyna_nel8; // #elements with 8 nodes (solids)
  int dyna_nel20; // # 20 node solid elements
  int dyna_nelth; // #thshells

  int dyna_nmmat;   // #mats
  int dyna_nummat2; // #mats for 1d/2d/3d/th elems
  int dyna_nummat4;
  int dyna_nummat8;
  int dyna_nummatth;

  int dyna_nv1d; // #vars for 1d/2d/3d/th elems
  int dyna_nv2d;
  int dyna_nv3d;
  int dyna_nv3dt;

  int dyna_maxint; // #layers of integration points
  int dyna_istrn; // indicates whether strain was written
  int dyna_neiph; // extra variables for solids
  int dyna_neips; // extra variables for shells

  int dyna_iu; // Indicators for: disp/vel/accel/temp
  int dyna_iv;
  int dyna_ia;
  int dyna_it;
  int dyna_idtdt; // temp change rate, numnp vals after temps

  int dyna_narbs; // dunno ... seems important

  int dyna_ioshl1; // 6 shell stresses
  int dyna_ioshl2; // shell plastic strain
  int dyna_ioshl3; // shell forces
  int dyna_ioshl4; // thick,energy,2 extra

  int dyna_extra; // double header length indicator
  int dyna_numprop; // number of properties dude!!!

  // just for checks ... can not be handled.
  int dyna_nmsph; // #nodes of sph
  int dyna_ngpsph; // #mats of sph
  int dyna_ialemat; // # some ale stuff .. it's late ...

  // Own Variables
  int nStates;
  vector<float> timesteps;

  bool own_nel10;
  bool own_external_numbers_I8 ;

  int wordPosition;
  int wordsToRead;
  int wordPositionStates;

  bool useFemzip;
  int femzip_state_offset;

  // Checks for already read variables
  bool plastic_strain_is_read;
  unsigned int plastic_strain_read;
  bool energy_is_read;
  unsigned int energy_read;
  bool strain_is_read;
  unsigned int strain_read;
  bool stress_is_read;
  unsigned int stress_read;
  bool disp_is_read;
  unsigned int disp_read;
  bool acc_is_read;
  unsigned int acc_read;
  bool vel_is_read;
  unsigned int vel_read;
  vector<unsigned int> history_is_read;
  vector<unsigned int> history_shell_is_read;
  vector<unsigned int> history_solid_is_read;
  
  vector<unsigned int> history_var_read;
  vector<unsigned int> history_shell_read;
  vector<unsigned int> history_solid_read;
  
  vector<unsigned int> history_var_mode;
  vector<unsigned int> history_shell_mode;
  vector<unsigned int> history_solid_mode;

  AbstractBuffer* buffer;
  DB_Nodes* db_nodes;
  DB_Parts* db_parts;
  DB_Elements* db_elements;

  // Functions
  void init_vars();
  void read_header();
  void read_geometry();
  vector<vector<float>> read_geometry_nodes();
  vector<vector<int>>   read_geometry_elem8();
  vector<vector<int>>   read_geometry_elem4();
  vector<vector<int>>   read_geometry_elem2();
  vector<vector<int>>   read_geometry_numbering();
  void                  read_geometry_parts();
  void read_states_init();
  void read_states_parse(vector<string>);
  unsigned int read_states_parse_readMode(string _variable);
  void read_states_displacement();
  void read_states_velocity();
  void read_states_acceleration();
  void read_states_elem8(unsigned int iState);
  void read_states_elem4(unsigned int iState);
  bool isFileEnding(int);

  // === P U B L I C === //
  public:
  D3plot (string filepath,bool _use_femzip,vector<string> _variables);
  ~D3plot();
  string get_filepath();
  void read_states(vector<string> _variables);
  vector<float> get_timesteps();
  DB_Nodes* get_db_nodes();
  DB_Parts* get_db_parts();
  DB_Elements* get_db_elements();
  bool displacement_is_read();

};

#endif
