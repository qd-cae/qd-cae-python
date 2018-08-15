
#ifndef D3PLOT_HPP
#define D3PLOT_HPP

// includes
#include <dyna_cpp/db/FEMFile.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace qd {

// forward declarations
class KeyFile;
class DB_Nodes;
class DB_Parts;
class DB_Elements;
class AbstractBuffer;

class D3plot : public FEMFile
{
private:
  std::string dyna_title;
  std::string dyna_datetime; // BUGGY

  int32_t dyna_filetype; // filetype, 1=d3plot, 5=d3part, 3=d3thdt

  int32_t dyna_ndim;   // dimension parameter
  int32_t dyna_icode;  // finite element code, should be 6
  int32_t dyna_numnp;  // number of nodes
  int32_t dyna_mdlopt; // describes element deletion
  int32_t dyna_mattyp; // material types section is read

  int32_t dyna_nglbv; // global vars per timestep

  int32_t dyna_nel2;  // #elements with 2 nodes (beams)
  int32_t dyna_nel4;  // #elements with 4 nodes (shells)
  int32_t dyna_nel48; // # 8 node shell elements?!?!?!
  int32_t dyna_nel8;  // #elements with 8 nodes (solids)
  int32_t dyna_nel20; // # 20 node solid elements
  int32_t dyna_nelth; // #thshells

  int32_t dyna_nmmat;   // #mats
  int32_t dyna_nummat2; // #mats for 1d/2d/3d/th elems
  int32_t dyna_nummat4;
  int32_t dyna_nummat8;
  int32_t dyna_nummatth;

  int32_t dyna_nv1d; // #vars for 1d/2d/3d/th elems
  int32_t dyna_nv2d;
  int32_t dyna_nv3d;
  int32_t dyna_nv3dt;

  int32_t dyna_maxint; // #layers of integration points
  int32_t dyna_istrn;  // indicates whether strain was written
  int32_t dyna_neiph;  // extra variables for solids
  int32_t dyna_neips;  // extra variables for shells
  int32_t dyna_neipb;  // extra variables for beams

  int32_t dyna_iu; // Indicators for: disp/vel/accel/temp
  int32_t dyna_iv;
  int32_t dyna_ia;
  int32_t dyna_it;
  int32_t dyna_idtdt; // temp change rate, numnp vals after temps

  int32_t dyna_narbs; // dunno ... seems important

  int32_t dyna_ioshl1; // 6 shell stresses
  int32_t dyna_ioshl2; // shell plastic strain
  int32_t dyna_ioshl3; // shell forces
  int32_t dyna_ioshl4; // thick,energy,2 extra

  int32_t dyna_extra;   // double header length indicator
  int32_t dyna_numprop; // number of properties dude!!!

  int32_t dyna_numrbe; // number of rigid body shell elems
  std::vector<int32_t>
    dyna_irbtyp; // rigid body material type numbers (internal)

  // just for checks ... can not be handled.
  int32_t dyna_nmsph;   // #nodes of sph
  int32_t dyna_ngpsph;  // #mats of sph
  int32_t dyna_ialemat; // # some ale stuff .. it's late ...

  // airbags
  int32_t dyna_npefg;              // something about airbag particles
  int32_t dyna_airbag_npartgas;    // number of airbags I think
  int32_t dyna_airbag_subver;      // who knows ... never explictly explained
  int32_t dyna_airbag_nchamber;    // number of chambers
  int32_t dyna_airbag_ngeom;       // number of geometry vars
  int32_t dyna_airbag_state_nvars; // number of state vars
  int32_t dyna_airbag_nparticles;  // number of particles
  int32_t dyna_airbag_state_geom;  // number of state geometry vars

  // Own Variables
  int32_t nStates;
  std::vector<float> timesteps;

  bool own_nel10;               // dunno anymore
  bool own_external_numbers_I8; // if 64bit integers written, not 32
  bool own_has_internal_energy;
  bool own_has_temperatures;
  bool own_has_mass_scaling_info; // true if dyna_it%10 != 0

  int32_t wordPosition; // tracker of word position in file
  int32_t wordsToRead;
  int32_t wordPositionStates; // remembers where states begin

  bool _is_femzipped; // femzip usage?
  int32_t femzip_state_offset;

  // Checks for already read variables
  bool plastic_strain_is_read;
  int32_t plastic_strain_read;
  bool energy_is_read;
  int32_t energy_read;
  bool strain_is_read;
  int32_t strain_read;
  bool stress_is_read;
  int32_t stress_read;
  bool stress_mises_is_read;
  int32_t stress_mises_read;
  bool disp_is_read;
  int32_t disp_read;
  bool acc_is_read;
  int32_t acc_read;
  bool vel_is_read;
  int32_t vel_read;

  std::vector<int32_t> history_is_read;
  std::vector<int32_t> history_shell_is_read;
  std::vector<int32_t> history_solid_is_read;

  std::vector<int32_t> history_var_read;
  std::vector<int32_t> history_shell_read;
  std::vector<int32_t> history_solid_read;

  std::vector<int32_t> history_var_mode;
  std::vector<int32_t> history_shell_mode;
  std::vector<int32_t> history_solid_mode;

  std::shared_ptr<AbstractBuffer> buffer;

  // header and metadata
  void read_header();
  void read_matsection();
  void read_airbag_section();

  // geometry reading
  void read_geometry();
  std::vector<std::vector<float>> read_geometry_nodes();
  std::vector<std::vector<int32_t>> read_geometry_elem8();
  std::vector<std::vector<int32_t>> read_geometry_elem4th();
  std::vector<std::vector<int32_t>> read_geometry_elem4();
  std::vector<std::vector<int32_t>> read_geometry_elem2();
  std::vector<std::vector<int32_t>> read_geometry_numbering();
  std::vector<int32_t> read_part_ids();
  void read_geometry_airbag();
  std::vector<std::string> read_part_names();

  // state reading
  void read_states_init();
  void read_states_parse(std::vector<std::string>);
  int32_t read_states_parse_readMode(const std::string& _variable) const;
  void read_states_displacement();
  void read_states_velocity();
  void read_states_acceleration();
  void read_states_elem8(size_t iState);
  void read_states_elem4(size_t iState);
  void read_states_elem4th(size_t iState);
  void read_states_airbag();
  bool isFileEnding(int32_t _iWord);

  // === P U B L I C === //
public:
  explicit D3plot(
    std::string filepath,
    std::vector<std::string> _variables = std::vector<std::string>(),
    bool use_femzip = false);
  explicit D3plot(std::string filepath,
                  std::string _variables = std::string(),
                  bool use_femzip = false);
  virtual ~D3plot();
  void info() const;
  void read_states(std::vector<std::string> _variables);
  void read_states(const std::string& _variable);
  void clear(
    const std::vector<std::string>& _variables = std::vector<std::string>());
  void clear(const std::string& _variable);
  size_t get_nTimesteps() const override;
  std::string get_title() const;
  std::vector<float> get_timesteps() const;
  /*
  void save_hdf5(const std::string& _filepath,
                 bool _overwrite_run,
                 const std::string& _run_name) const;
  */

  bool displacement_is_read() const;
};

/** Tells whether displacements were loaded.
 *
 * @return disp_is_read : boolean whether the disp was read
 */
inline bool
D3plot::displacement_is_read() const
{
  return this->disp_is_read;
}

/** Get the number of states in the file
 *
 * @return nStates
 */
inline size_t
D3plot::get_nTimesteps() const
{
  return this->timesteps.size();
}

} // namespace qd

#endif
