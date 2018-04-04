
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
  std::string title;
  std::string datetime; // missing

  int32_t ndim;   // dimension parameter
  int32_t icode;  // finite element code, should be 6
  int32_t numnp;  // number of nodes
  int32_t mdlopt; // describes element deletion
  int32_t mattyp; // material types section is read

  int32_t nglbv; // global vars per timestep

  int32_t nel2;  // #elements with 2 nodes (beams)
  int32_t nel4;  // #elements with 4 nodes (shells)
  int32_t nel48; // # 8 node shell elements?!?!?!
  int32_t nel8;  // #elements with 8 nodes (solids)
  int32_t nel20; // # 20 node solid elements
  int32_t nelth; // #thshells

  int32_t nmmat;   // #mats
  int32_t nummat2; // #mats for 1d/2d/3d/th elems
  int32_t nummat4;
  int32_t nummat8;
  int32_t nummatth;

  int32_t nv1d; // #vars for 1d/2d/3d/th elems
  int32_t nv2d;
  int32_t nv3d;
  int32_t nv3dt;

  int32_t maxint; // #layers of integration points
  int32_t istrn;  // indicates whether strain was written
  int32_t neiph;  // extra variables for solids
  int32_t neips;  // extra variables for shells
  int32_t neipb;  // extra variables for beams

  int32_t iu; // Indicators for: disp/vel/accel/temp
  int32_t iv;
  int32_t ia;
  int32_t it;
  int32_t idtdt; // temp change rate, numnp vals after temps

  int32_t narbs; // dunno ... seems important

  int32_t ioshl1; // 6 shell stresses
  int32_t ioshl2; // shell plastic strain
  int32_t ioshl3; // shell forces
  int32_t ioshl4; // thick,energy,2 extra

  int32_t extra;   // double header length indicator
  int32_t numprop; // number of properties dude!!!

  int32_t numrbe;         // number of rigid body shell elems
  Tensor<int32_t> irbtyp; // rigid body material type numbers (internal)

  // just for checks ... can not be handled.
  int32_t nmsph;   // #nodes of sph
  int32_t ngpsph;  // #mats of sph
  int32_t ialemat; // # some ale stuff .. it's late ...

  // airbags
  int32_t npefg;                     // something about airbag particles
  int32_t airbag_npartgas;           // number of airbags I think
  int32_t airbag_subver;             // who knows ... never explictly explained
  int32_t airbag_nchamber;           // number of chambers
  int32_t airbag_ngeom;              // number of geometry vars
  int32_t airbag_state_nvars;        // number of state vars
  int32_t airbag_nparticles;         // number of particles
  int32_t airbag_state_geom;         // number of state geometry vars
  std::vector<int32_t> airbag_nlist; // type of airbag var (1=int, 2=float)

  D3plotHeader();
};

} // namespace:qd
#endif