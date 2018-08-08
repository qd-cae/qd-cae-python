
#ifndef RAWD3PLOT_HPP
#define RAWD3PLOT_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <dyna_cpp/math/Tensor.hpp>

namespace qd {

class RawD3plot
{

private:
  // Dyna file variables
  std::string dyna_title;
  std::string dyna_datetime; // missing

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
  std::shared_ptr<Tensor<int32_t>>
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
  std::vector<int32_t> dyna_airbag_nlist; // type of airbag var (1=int, 2=float)

  // Own Variables
  int32_t nStates;
  std::vector<float> timesteps;

  bool own_nel10;               // dunno anymore
  bool own_external_numbers_I8; // if 64bit integers written, not 32
  bool own_has_internal_energy;
  bool own_has_temperatures;
  bool
    own_has_mass_scaling_info; // true if dyna_it > 10 (little more complicate)

  int32_t own_nDeletionVars;

  int32_t wordPosition; // tracker of word position in file
  int32_t wordsToRead;
  int32_t wordPositionStates; // remembers where states begin

  bool _is_femzipped; // femzip usage?
  int32_t femzip_state_offset;

  std::shared_ptr<AbstractBuffer> buffer;

  // Data
  std::map<std::string, std::shared_ptr<Tensor<int32_t>>> int_data;
  std::map<std::string, std::shared_ptr<Tensor<float>>> float_data;
  std::map<std::string, std::vector<std::string>> string_data;

  // header and metadata
  void read_header();
  void read_matsection();
  void read_airbag_section();

  // geometry reading
  void read_geometry();
  void read_geometry_nodes();
  void read_geometry_elem8();
  void read_geometry_elem4th();
  void read_geometry_elem4();
  void read_geometry_elem2();
  void read_geometry_numbering();
  void read_geometry_airbag();
  void read_part_ids();
  void read_part_names();

  // state reading
  void read_states();
  void read_states_nodes_mass_scaling();
  void read_states_displacement();
  void read_states_velocity();
  void read_states_acceleration();
  void read_states_elem8();
  void read_states_elem4();
  void read_states_elem4th();
  void read_states_elem2();
  void read_states_elem_deletion();
  void read_states_airbag();
  bool isFileEnding(int32_t _iWord);

  // === P U B L I C === //
public:
  explicit RawD3plot();
  explicit RawD3plot(std::string filepath, bool use_femzip = false);
  virtual ~RawD3plot();

  // disallow copy
  RawD3plot(const RawD3plot&) = delete;
  RawD3plot& operator=(const RawD3plot&) = delete;

  void info() const;
  std::string get_title() const;

  Tensor_ptr<int32_t> get_int_data(const std::string& _name);
  std::vector<std::string> get_int_names() const;
  std::vector<std::string> get_string_data(const std::string& _name);
  std::vector<std::string> get_string_names() const;
  Tensor_ptr<float> get_float_data(const std::string& _name);
  std::vector<std::string> get_float_names() const;
  void set_float_data(const std::string& _name,
                      std::vector<size_t> _shape,
                      const float* _data_ptr);
  void set_int_data(const std::string& _name,
                    std::vector<size_t> _shape,
                    const int* _data_ptr);
  void set_int_data(const std::string& _name, Tensor_ptr<int32_t> _data);
  void set_string_data(const std::string& _name,
                       const std::vector<std::string>& _data);
};

} // namespace std

#endif