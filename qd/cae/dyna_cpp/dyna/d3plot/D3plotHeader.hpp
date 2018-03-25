
#ifndef D3PLOTHEADER_HPP
#define D3PLOTHEADER_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <dyna_cpp/math/Tensor.hpp>

namespace qd {

class D3plotHeader
{
public:
  // Dyna file variables
  std::string dyna_title;
  std::string dyna_datetime; // missing

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

  int32_t dyna_numrbe;         // number of rigid body shell elems
  Tensor<int32_t> dyna_irbtyp; // rigid body material type numbers (internal)

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

  D3plotHeader();
};

} // namespace:qd
#endif