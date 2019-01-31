
#include <cmath>
#include <string>

#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/db/Part.hpp>
#include <dyna_cpp/dyna/d3plot/D3plot.hpp>
#include <dyna_cpp/dyna/d3plot/D3plotBuffer.hpp>
#include <dyna_cpp/utility/FEM_Utility.hpp>
#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/MathUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

#ifdef QD_USE_FEMZIP
#include "FemzipBuffer.hpp"
#endif

#ifdef QD_USE_HDF5
#include <H5Cpp.h>
#include <dyna_cpp/utility/HDF5_Utility.hpp>
#endif

namespace qd {

/** Constructor for a D3plot.
 * @param filepath : path to the d3plot file
 * @param state_variables : which state variables to read, see member function
 *                          read_states
 * @param use_femzip : set to true if your d3plot was femzipped
 */
D3plot::D3plot(std::string _filename,
               std::vector<std::string> _state_variables,
               bool use_femzip)
  : FEMFile(_filename)
  , dyna_filetype(-1)
  , dyna_ndim(-1)
  , dyna_icode(-1)
  , dyna_numnp(-1)
  , dyna_mdlopt(-1)
  , dyna_mattyp(-1)
  , dyna_nglbv(-1)
  , dyna_nel2(-1)
  , dyna_nel4(-1)
  , dyna_nel48(-1)
  , dyna_nel8(-1)
  , dyna_nel20(-1)
  , dyna_nelth(-1)
  , dyna_nmmat(-1)
  , dyna_nummat2(-1)
  , dyna_nummat4(-1)
  , dyna_nummat8(-1)
  , dyna_nummatth(-1)
  , dyna_nv1d(-1)
  , dyna_nv2d(-1)
  , dyna_nv3d(-1)
  , dyna_nv3dt(-1)
  , dyna_maxint(-1)
  , dyna_istrn(-1)
  , dyna_neiph(-1)
  , dyna_neips(-1)
  , dyna_neipb(-1)
  , dyna_iu(-1)
  , dyna_iv(-1)
  , dyna_ia(-1)
  , dyna_it(-1)
  , dyna_idtdt(-1)
  , dyna_narbs(-1)
  , dyna_ioshl1(-1)
  , dyna_ioshl2(-1)
  , dyna_ioshl3(-1)
  , dyna_ioshl4(-1)
  , dyna_extra(-1)
  , dyna_numprop(-1)
  , dyna_numrbe(-1)
  , dyna_nmsph(-1)
  , dyna_ngpsph(-1)
  , dyna_ialemat(-1)
  , dyna_npefg(-1)
  , dyna_airbag_npartgas(-1)
  , dyna_airbag_subver(-1)
  , dyna_airbag_nchamber(-1)
  , dyna_airbag_ngeom(-1)
  , dyna_airbag_state_nvars(-1)
  , dyna_airbag_nparticles(-1)
  , dyna_airbag_state_geom(-1)
  , nStates(0)
  , own_nel10(false)
  , own_external_numbers_I8(false)
  , own_has_internal_energy(false)
  , own_has_temperatures(false)
  , own_has_mass_scaling_info(false)
  , wordPosition(0)
  , wordsToRead(0)
  , wordPositionStates(0)
  , _is_femzipped(false)
  , femzip_state_offset(0)
  , plastic_strain_is_read(false)
  , plastic_strain_read(0)
  , energy_is_read(false)
  , energy_read(0)
  , strain_is_read(false)
  , strain_read(0)
  , stress_is_read(false)
  , stress_read(0)
  , stress_mises_is_read(false)
  , stress_mises_read(0)
  , disp_is_read(false)
  , disp_read(0)
  , acc_is_read(false)
  , acc_read(0)
  , vel_is_read(false)
  , vel_read(0)
  , buffer(nullptr)
{
// check for femzip
#ifdef QD_USE_FEMZIP
  // _is_femzipped = FemzipBuffer::is_femzipped(_filename);
  _is_femzipped = use_femzip;
  if (_is_femzipped) {
    buffer = std::make_shared<FemzipBuffer>(_filename);
  } else {
    constexpr int32_t bytesPerWord = 4;
    buffer = std::make_shared<D3plotBuffer>(_filename, bytesPerWord);
  }
#else
  if (use_femzip)
    throw(
      std::invalid_argument("Library was compiled without femzip support."));

  const int32_t bytesPerWord = 4;
  buffer = std::make_shared<D3plotBuffer>(_filename, bytesPerWord);
#endif

  this->buffer->read_geometryBuffer(); // deallocated in read_geometry

  // Header + Geometry
  this->read_header();
  this->read_matsection();
  this->read_airbag_section();
  this->read_geometry();

  // States
  //
  // This routine must run through, even though no variables might be read.
  // This is due to the fact, that femzip must read the states before
  // closing the file. It is not possible to leave this out.
  //
  // Need to check this but I think I also set some other vars like the
  // beginning of the state section and the state count.
  this->read_states(_state_variables);
}

/** Constructor for a D3plot.
 * @param filepath : path to the d3plot file
 * @param _variable : which state variable to read, see member function
 *                    read_states
 * @param use_femzip : set to true if your d3plot was femzipped
 */
D3plot::D3plot(std::string _filepath, std::string _variable, bool use_femzip)
  : D3plot(_filepath,
           [_variable](std::string) -> std::vector<std::string> {
             if (_variable.empty()) {
               return std::vector<std::string>();
             } else {
               std::vector<std::string> vec = { _variable };
               return vec;
             }
           }(_variable),
           use_femzip)
{}

/*
 * Destructor
 *
 */
D3plot::~D3plot()
{
#ifdef QD_DEBUG
  std::cout << "D3plot::~D3plot() called." << std::endl;
#endif
}

/*
 * Read the header data.
 *
 */
void
D3plot::read_header()
{
#ifdef QD_DEBUG
  std::cout << "> HEADER " << std::endl;
#endif

  dyna_filetype = this->buffer->read_int(11);
  if (dyna_filetype > 1000) {
    dyna_filetype -= 1000;
    own_external_numbers_I8 = true;
  }
  if ((dyna_filetype != 0) && (dyna_filetype != 1) && (dyna_filetype != 5)) {
    throw(std::runtime_error(
      "Wrong filetype " + std::to_string(this->buffer->read_int(11)) +
      " != 1 (or 5) in header of d3plot. Your file might be in Double "
      "Precision or the endian of the file is not the endian of the "
      "machine."));
  }

  this->dyna_title = this->buffer->read_str(0, 10);

  // BUGGY
  // bugfix: its a string not int32_t
  /*
  int32_t timestamp = this->buffer->read_int(10);
  time_t timestamp2 = (time_t) timestamp;
  this->dyna_datetime = asctime(localtime(&timestamp2));
  std::cout << timestamp << std::endl;
  std::cout << this->dyna_datetime << std::endl;
  */

  this->dyna_ndim = this->buffer->read_int(15);
  this->dyna_mattyp = 0;
  if ((this->dyna_ndim == 5) | (this->dyna_ndim == 7)) {
    // connectivities are unpacked?
    this->dyna_mattyp = 1;
    this->dyna_ndim = 3;
  } else if (this->dyna_ndim == 4) {
    // connectivities are unpacked?
    this->dyna_ndim = 3;
  } else if (this->dyna_ndim > 5) {
    throw(std::runtime_error(
      "State data contains rigid road surface, which can not be handled."));
  } else {
    throw(std::runtime_error("Invalid parameter dyna_ndim=" +
                             std::to_string(this->dyna_ndim)));
  }

  this->dyna_numnp = this->buffer->read_int(16);
  this->dyna_icode = this->buffer->read_int(17);
  this->dyna_nglbv = this->buffer->read_int(18);

  this->dyna_iu = this->buffer->read_int(20);
  this->dyna_iv = this->buffer->read_int(21);
  this->dyna_ia = this->buffer->read_int(22);
  this->dyna_it = this->buffer->read_int(19);
  if (this->dyna_it != 0) {
    own_has_temperatures = (this->dyna_it % 10) != 0;
  }
  if (this->dyna_it >= 10) {
    own_has_mass_scaling_info = (this->dyna_it / 10) == 1;
  }

  this->dyna_nel2 = this->buffer->read_int(28);
  this->dyna_nel4 = this->buffer->read_int(31);
  this->dyna_nel8 = this->buffer->read_int(23);
  this->dyna_nelth = this->buffer->read_int(40);
  this->dyna_nel48 = this->buffer->read_int(55);
  if (this->dyna_nel8 < 0) {
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
  if (this->dyna_maxint >= 0) {
    this->dyna_mdlopt = 0;
  } else if (this->dyna_maxint < 0) {
    this->dyna_mdlopt = 1;
    this->dyna_maxint = abs(this->dyna_maxint);
  }
  if (this->dyna_maxint > 10000) {
    this->dyna_mdlopt = 2;
    this->dyna_maxint = abs(this->dyna_maxint) - 10000;
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
  this->dyna_npefg = this->buffer->read_int(54);

  // Header extra!
  if (this->dyna_extra > 0) {
    this->dyna_nel20 = this->buffer->read_int(64);
    this->dyna_neipb = this->buffer->read_int(67);
  } else {
    this->dyna_nel20 = 0;
  }

  // istrn in idtdt
  if (this->dyna_idtdt > 100) {
    this->dyna_istrn = this->dyna_idtdt % 10000;

    // istrn needs to be calculated
  } else {
    if (dyna_nv2d > 0) {
      if (dyna_nv2d -
            dyna_maxint * (6 * dyna_ioshl1 + dyna_ioshl2 + dyna_neips) +
            8 * dyna_ioshl3 + 4 * dyna_ioshl4 >
          1) {
        dyna_istrn = 1;
      } else {
        dyna_istrn = 0;
      }
    } else if (dyna_nelth > 0) {
      if (dyna_nv3dt -
            dyna_maxint * (6 * dyna_ioshl1 + dyna_ioshl2 + dyna_neips) >
          1) {
        dyna_istrn = 1;
      } else {
        dyna_istrn = 0;
      }
    }
  }

  // check for shell internal energy
  int32_t shell_vars_behind_layers =
    dyna_nv2d - dyna_maxint * (6 * dyna_ioshl1 + dyna_ioshl2 + dyna_neips) +
    8 * dyna_ioshl3 + 4 * dyna_ioshl4;

  if (dyna_istrn == 0) {

    // no strain, but energy
    if (shell_vars_behind_layers > 1 && shell_vars_behind_layers < 6) {
      own_has_internal_energy = true;
    } else {
      own_has_internal_energy = false;
    }

  } else if (dyna_istrn == 1) {

    // strain and energy
    if (shell_vars_behind_layers > 12) {
      own_has_internal_energy = true;
    } else {
      own_has_internal_energy = false;
    }
  }

#ifdef QD_DEBUG
  this->info();
#endif

  /* === CHECKS === */
  // sph
  if ((this->dyna_nmsph != 0) | (this->dyna_ngpsph != 0))
    throw(std::runtime_error("SPH mats and elements can not be handled."));
  // ale
  if (this->dyna_ialemat != 0)
    throw(std::runtime_error("ALE can not be handled."));
  // no temps
  if (own_has_temperatures)
    throw(std::runtime_error("Can not handle temperatures in file."));
  if (own_external_numbers_I8)
    throw(
      std::runtime_error("Can not handle external ids with 8 byte length."));

  // update position
  if (this->dyna_extra > 0) {
    wordPosition = 64 * 2; // header has 128 words
  } else {
    wordPosition = 64; // header has 64 words
  }
}

/**  Print info about the data in the d3plot to the console.
 *
 */
void
D3plot::info() const
{
  std::cout << "Title:  " << this->dyna_title << '\n';
  std::cout << "nNodes : " << this->dyna_numnp << '\n';
  std::cout << "nElem2 : " << this->dyna_nel2 << '\n';
  std::cout << "nElem4 : " << this->dyna_nel4 << '\n';
  std::cout << "nElem8 : " << this->dyna_nel8 << '\n';
  std::cout << "nElem20: " << this->dyna_nel20 << '\n';
  std::cout << "nElemTh: " << this->dyna_nelth << '\n';
  std::cout << "nElem48: " << this->dyna_nel48 << '\n';
  std::cout << "nMats-Solver: " << this->dyna_nmmat << '\n';
  std::cout << "nMats-Input : "
            << this->dyna_nummat2 + this->dyna_nummat4 + this->dyna_nummat8 +
                 this->dyna_nummatth
            << '\n';
  std::cout << "nMat2 : " << this->dyna_nummat2 << '\n';
  std::cout << "nMat4 : " << this->dyna_nummat4 << '\n';
  std::cout << "nMat8 : " << this->dyna_nummat8 << '\n';
  std::cout << "nMatTh: " << this->dyna_nummatth << '\n';
  std::cout << "disp : " << this->dyna_iu << '\n';
  std::cout << "vel  : " << this->dyna_iv << '\n';
  std::cout << "accel: " << this->dyna_ia << '\n';
  std::cout << "temp : " << this->dyna_it << '\n';
  std::cout << "shell-stress: " << this->dyna_ioshl1 << '\n';
  std::cout << "shell-plstrn: " << this->dyna_ioshl2 << '\n';
  std::cout << "shell-forces: " << this->dyna_ioshl3 << '\n';
  std::cout << "shell-stuff : " << this->dyna_ioshl4 << '\n';
  std::cout << "shell-strain: " << this->dyna_istrn << '\n';
  std::cout << "shell-nInteg: " << this->dyna_maxint << '\n';
  std::cout << "shell-nHists: " << this->dyna_neiph << '\n';
  std::cout << "nVar1D : " << this->dyna_nv1d << '\n';
  std::cout << "nVar2D : " << this->dyna_nv2d << '\n';
  std::cout << "nVar3D : " << this->dyna_nv3d << '\n';
  std::cout << "nVar3DT: " << this->dyna_nv3dt << '\n';
  std::cout << "state-globals: " << this->dyna_nglbv << std::endl;
#ifdef QD_DEBUG
  std::cout << "icode: " << this->dyna_icode << " (solver code flag)\n"
            << "neips: " << this->dyna_neips << " (additional solid vars)\n"
            << "neiph: " << this->dyna_neiph << " (additional shell vars)\n"
            << "mattyp: " << this->dyna_mattyp << "\n"
            << "narbs: " << this->dyna_narbs << "\n"
            << "idtdt: " << this->dyna_idtdt << "\n"
            << "npefg: " << this->dyna_npefg << " (airbag particles)\n"
            << "extra: " << this->dyna_extra << '\n'
            << "node_mass_scaling: " << own_has_mass_scaling_info << std::endl;
#endif
}

/** Read the rigid body material section.
 *
 * Does nothing if mattyp==0.
 */
void
D3plot::read_matsection()
{

#ifdef QD_DEBUG
  std::cout << "> MATSECTION"
            << "\n"
            << "at word " << this->wordPosition << std::endl;
#endif

  // Nothing to do
  if (this->dyna_mattyp == 0) {
    dyna_numrbe = 0;
    return;
  }

  this->dyna_numrbe = this->buffer->read_int(wordPosition); // rigid shells
  int32_t tmp_nummat = this->buffer->read_int(wordPosition + 1);
#ifdef QD_DEBUG
  std::cout << "nummat=" << tmp_nummat << "\n"
            << "numrbe=" << this->dyna_numrbe << std::endl;
#endif

  if (tmp_nummat != dyna_nmmat) {
#ifdef QD_DEBUG
    std::cout << "nmmat=" << this->dyna_nmmat << std::endl;
#endif
    throw(std::runtime_error("dyna_nmmat != nummat in matsection!"));
  }

  int32_t start = wordPosition + 2;
  int32_t end = start + tmp_nummat;
  this->dyna_irbtyp.reserve(tmp_nummat);
  for (int32_t iPosition = start; iPosition < end; ++iPosition) {
    this->dyna_irbtyp.push_back(this->buffer->read_int(iPosition));
  }

  this->wordPosition += 2 + tmp_nummat;
}

/** Read the airbag section data
 *
 * If airbags are used, then we need to skip the airbag data blocks.
 */
void
D3plot::read_airbag_section()
{

#ifdef QD_DEBUG
  std::cout << "> AIRBAG (PARTICLES)" << std::endl;
#endif

  // skip airbag particle section
  if ((this->dyna_npefg > 0) && (this->dyna_npefg < 10000000)) {

    this->dyna_airbag_npartgas = this->dyna_npefg % 1000; // nAirbags?!
    this->dyna_airbag_subver = this->dyna_npefg / 1000;   // n

    this->dyna_airbag_ngeom = this->buffer->read_int(wordPosition++);
    this->dyna_airbag_state_nvars = this->buffer->read_int(wordPosition++);
    this->dyna_airbag_nparticles = this->buffer->read_int(wordPosition++);
    this->dyna_airbag_state_geom = this->buffer->read_int(wordPosition++);
    this->dyna_airbag_nchamber = 0;
    if (dyna_airbag_subver == 4) {
      this->dyna_airbag_nchamber = this->buffer->read_int(wordPosition++);
    }

    int32_t dyna_airbag_nlist =
      dyna_airbag_ngeom + dyna_airbag_state_nvars + dyna_airbag_state_geom;

    wordPosition += dyna_airbag_nlist; // type of each variable (1=int, 2=float)
    wordPosition += 8 * dyna_airbag_nlist; // 8 char variable names

#ifdef QD_DEBUG
    std::cout << "dyna_airbag_npartgas: " << this->dyna_airbag_npartgas << "\n"
              << "dyna_airbag_subver: " << this->dyna_airbag_subver << "\n"
              << "dyna_airbag_ngeom: " << this->dyna_airbag_ngeom << "\n"
              << "dyna_airbag_nparticles: " << this->dyna_airbag_nparticles
              << "\n"
              << "dyna_airbag_state_nvars: " << this->dyna_airbag_state_nvars
              << "\n"
              << "dyna_airbag_state_geom: " << this->dyna_airbag_state_geom
              << "\n"
              << "dyna_airbag_nchamber: " << this->dyna_airbag_nchamber << "\n"
              << "dyna_airbag_nlist: " << dyna_airbag_nlist << std::endl;
#endif
  } else {
    this->dyna_airbag_npartgas = 0;
    this->dyna_airbag_subver = 0;
    this->dyna_airbag_ngeom = 0;
    this->dyna_airbag_state_nvars = 0;
    this->dyna_airbag_nparticles = 0;
    this->dyna_airbag_state_geom = 0;
  }
}

/** Read the geometry mesh (after the header)
 */
void
D3plot::read_geometry()
{
#ifdef QD_DEBUG
  std::cout << "> GEOMETRY" << std::endl;
#endif

  /* === NODES === */
  auto buffer_nodes = this->read_geometry_nodes();

  /* === ELEMENTS === */
  // Order MATTERS, do not swap routines.

  // 8-Node Solids
  auto buffer_elems8 = this->read_geometry_elem8();

  // 8-Node Thick Shells
  auto buffer_elems4th = read_geometry_elem4th();

  // 2-Node Beams
  auto buffer_elems2 = this->read_geometry_elem2();

  // 4-Node Elements
  auto buffer_elems4 = this->read_geometry_elem4();

  /* === NUMBERING === */
  auto buffer_numbering = this->read_geometry_numbering();

  auto part_numbering = read_part_ids();

  /* === AIRBAGS === */
  this->read_geometry_airbag();

  // Check if part names are here
  std::vector<std::string> part_names;
  // if (!isFileEnding(wordPosition)) {
  if (isFileEnding(wordPosition)) {
#ifdef QD_DEBUG
    std::cout << "At word position: " << wordPosition << std::endl;
#endif
    // throw(
    //   std::runtime_error("Anticipated file ending wrong in geometry
    //   section."));
    // }
    wordPosition++;

    /* === PARTS === */
    this->buffer->free_geometryBuffer();
    this->buffer->read_partBuffer();
    if (this->_is_femzipped)
      wordPosition = 1; // don't ask me why not 0 ...

    part_names = this->read_part_names(); // directly creates parts

    if (!isFileEnding(wordPosition)) {
#ifdef QD_DEBUG
      std::cout << "At word position: " << wordPosition << std::endl;
#endif
      throw(
        std::runtime_error("Anticipated file ending wrong in part section."));
    }

    this->buffer->free_partBuffer();
  }

/* ====== D A T A B A S E S ====== */

// Parts
#ifdef QD_DEBUG
  std::cout << "Adding parts ... ";
#endif
  auto* db_parts = this->get_db_parts();
  for (size_t i_part = 0; i_part < part_numbering.size(); ++i_part) {

    auto part = db_parts->add_partByID(part_numbering[i_part]);
    if (i_part < part_names.size())
      part->set_name(part_names[i_part]);
  }
#ifdef QD_DEBUG
  std::cout << this->get_db_parts()->get_nParts() << " done." << std::endl;
#endif

// Nodes
#ifdef QD_DEBUG
  std::cout << "Adding nodes ... ";
#endif
  if (buffer_numbering[0].size() != buffer_nodes.size())
    throw(std::runtime_error(
      "Buffer node-numbering and buffer-nodes have different sizes."));
  DB_Nodes* db_nodes = this->get_db_nodes();
  db_nodes->reserve(buffer_nodes.size());
  for (size_t ii = 0; ii < buffer_nodes.size(); ii++) {
    db_nodes->add_node(buffer_numbering[0][ii], buffer_nodes[ii]);
  }
#ifdef QD_DEBUG
  std::cout << this->get_db_nodes()->get_nNodes() << " done." << std::endl;
#endif

  DB_Elements* db_elems = this->get_db_elements();

// Beams
#ifdef QD_DEBUG
  std::cout << "Adding beams ... ";
#endif
  db_elems->reserve(Element::BEAM, buffer_elems2.size());

  const auto buffer_elems2_size = static_cast<int64_t>(buffer_elems2.size());
#pragma omp parallel for schedule(dynamic)
  for (int64_t ii = 0; ii < buffer_elems2_size; ++ii) {
    db_elems->add_element_byD3plot(
      Element::BEAM, buffer_numbering[2][ii], buffer_elems2[ii]);
  }
#ifdef QD_DEBUG
  std::cout << this->get_db_elements()->get_nElements(Element::BEAM) << " done."
            << std::endl;
#endif

// Shells
#ifdef QD_DEBUG
  std::cout << "Adding shells ... ";
#endif
  int32_t nRigidShells = 0;
  db_elems->reserve(Element::SHELL, buffer_elems4.size());

  const auto buffer_elems4_size = static_cast<int64_t>(buffer_elems4.size());
#pragma omp parallel for schedule(dynamic)
  for (int64_t ii = 0; ii < buffer_elems4_size; ++ii) {
    // for (size_t ii = 0; ii < buffer_elems4.size(); ++ii) {

    auto elem = db_elems->add_element_byD3plot(
      Element::SHELL, buffer_numbering[3][ii], buffer_elems4[ii]);

    // check if rigid material, very complicated ...
    // this bug took me 3 Days! material indexes start again at 1, not 0 :(
    if ((dyna_mattyp == 1) &&
        (this->dyna_irbtyp[buffer_elems4[ii].back() - 1] == 20)) {
      elem->set_is_rigid(true);
      ++nRigidShells;
    }
  }
#ifdef QD_DEBUG
  std::cout << this->get_db_elements()->get_nElements(Element::SHELL)
            << " done." << std::endl;
#endif
  // if (dyna_filetype == 1 && nRigidShells != this->dyna_numrbe)
  // throw(std::runtime_error(
  //   "nRigidShells != numrbe: " + std::to_string(nRigidShells) +
  //   " != " + std::to_string(this->dyna_numrbe)));
  // this->dyna_numrbe = nRigidShells;

// Solids
#ifdef QD_DEBUG
  std::cout << "Adding solids ... ";
#endif
  db_elems->reserve(Element::SOLID, buffer_elems8.size());

  const auto buffer_elems8_size = static_cast<int64_t>(buffer_elems8.size());
#pragma omp parallel for schedule(dynamic)
  for (int64_t ii = 0; ii < buffer_elems8_size; ++ii) {
    db_elems->add_element_byD3plot(
      Element::SOLID, buffer_numbering[1][ii], buffer_elems8[ii]);
  }
#ifdef QD_DEBUG
  std::cout << get_db_elements()->get_nElements(Element::SOLID) << " done."
            << std::endl;
#endif

// Thick Shells
#ifdef QD_DEBUG
  std::cout << "Adding thick shells ... ";
#endif
  db_elems->reserve(Element::TSHELL, buffer_elems4th.size());

  const auto buffer_elems4th_size =
    static_cast<int64_t>(buffer_elems4th.size());
#pragma omp parallel for schedule(dynamic)
  for (int64_t ii = 0; ii < buffer_elems4th_size; ++ii) {
    db_elems->add_element_byD3plot(
      Element::TSHELL, buffer_numbering[4][ii], buffer_elems4th[ii]);
  }
#ifdef QD_DEBUG
  std::cout << get_db_elements()->get_nElements(Element::TSHELL) << " done."
            << std::endl;
#endif
}

/*
 * Read the nodes in the geometry section.
 *
 */
std::vector<std::vector<float>>
D3plot::read_geometry_nodes()
{
#ifdef QD_DEBUG
  std::cout << "Reading nodes at word " << wordPosition << " ... ";
#endif

  wordsToRead = dyna_numnp * dyna_ndim;
  std::vector<std::vector<float>> buffer_nodes(dyna_numnp,
                                               std::vector<float>(3));

  size_t jj = 0;
  for (int32_t ii = wordPosition; ii < wordPosition + wordsToRead;
       ii += 3, ++jj) {
    buffer->read_float_array(ii, 3, buffer_nodes[jj]);
  }

  // Update word position
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  return buffer_nodes;
}

/** Read the 8 noded elements in the geometry section.
 *
 * @return buffer_elems8 : element data buffer
 *
 */
std::vector<std::vector<int32_t>>
D3plot::read_geometry_elem8()
{
  // Check
  if (dyna_nel8 == 0)
    return std::vector<std::vector<int32_t>>();

#ifdef QD_DEBUG
  std::cout << "Reading solids at word " << wordPosition << " ... ";
#endif

  // currently each element has 8 nodes-ids and 1 mat-id
  const int32_t nVarsElem8 = 9;

  // allocate
  std::vector<std::vector<int32_t>> buffer_elems8(
    dyna_nel8, std::vector<int32_t>(nVarsElem8));

  wordsToRead = nVarsElem8 * dyna_nel8;
  size_t iElement = 0;
  size_t iData = 0;
  // Loop over elements
  for (int32_t ii = wordPosition; ii < wordPosition + wordsToRead;
       ii += nVarsElem8) {
    // Loop over element data
    iData = 0;
    for (int32_t jj = ii; jj < ii + nVarsElem8; jj++) {
      buffer_elems8[iElement][iData] = buffer->read_int(jj);
      iData++;
    }

    iElement++;
  }

  // Update word position
  wordPosition += wordsToRead;
  if (own_nel10)
    wordPosition += 2 * dyna_nel8;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  return std::move(buffer_elems8);
}

/*
 * Read the 4 noded elements in the geometry section.
 *
 */
std::vector<std::vector<int32_t>>
D3plot::read_geometry_elem4()
{
  // Check
  if (dyna_nel4 == 0)
    return std::vector<std::vector<int32_t>>();

#ifdef QD_DEBUG
  std::cout << "Reading shells at word " << wordPosition << " ... ";
#endif

  const int32_t nVarsElem4 = 5;

  // allocate
  std::vector<std::vector<int32_t>> buffer_elems4(
    dyna_nel4, std::vector<int32_t>(nVarsElem4));

  wordsToRead = nVarsElem4 * dyna_nel4;
  size_t iElement = 0;
  size_t iData = 0;
  // Loop over elements
  for (int32_t ii = wordPosition; ii < wordPosition + wordsToRead;
       ii += nVarsElem4) {
    // Loop over element data
    iData = 0;
    for (int32_t jj = ii; jj < ii + nVarsElem4; ++jj) {
      buffer_elems4[iElement][iData] = buffer->read_int(jj);
      ++iData;
    }
    ++iElement;
  }

  // Update word position
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  return std::move(buffer_elems4);
}

/*
 * Read the 2 noded elements in the geometry section.
 *
 */
std::vector<std::vector<int32_t>>
D3plot::read_geometry_elem2()
{
  // Check
  if (dyna_nel2 == 0)
    return std::vector<std::vector<int32_t>>();

#ifdef QD_DEBUG
  std::cout << "Reading beams at word " << wordPosition << " ... ";
#endif

  const int32_t nVarsElem2 = 6;

  // allocate
  std::vector<std::vector<int32_t>> buffer_elems2(dyna_nel2,
                                                  std::vector<int32_t>(3));

  wordsToRead = nVarsElem2 * dyna_nel2;
  int32_t iElement = 0;
  // Loop over elements
  for (int32_t ii = wordPosition; ii < wordPosition + wordsToRead;
       ii += nVarsElem2) {
    // Loop over element data
    /*
    iData = 0;
    std::vector<int32_t> elemData(nVarsElem2);
    for(int32_t jj=ii;jj<ii+nVarsElem2;jj++){
      elemData[iData] = buffer->read_int(jj);
      iData++;
    }
    */
    buffer_elems2[iElement][0] = buffer->read_int(ii);
    buffer_elems2[iElement][1] = buffer->read_int(ii + 1);
    buffer_elems2[iElement][2] = buffer->read_int(ii + 5); // mat

    iElement++;
  }

  // Update word position
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  return std::move(buffer_elems2);
}

/** Read the thick shell data from the geometry section
 *
 * @return buffer_elems4th : element data buffer
 *
 */
std::vector<std::vector<int32_t>>
D3plot::read_geometry_elem4th()
{
  // Check
  if (dyna_nelth == 0)
    return std::vector<std::vector<int32_t>>();

#ifdef QD_DEBUG
  std::cout << "Reading thick shells at word " << wordPosition << " ... ";
#endif

  // 8 nodes and material id
  const int32_t nVarsElem4th = 9;

  // allocate
  std::vector<std::vector<int32_t>> buffer_elems4th(
    dyna_nelth, std::vector<int32_t>(nVarsElem4th));

  wordsToRead = nVarsElem4th * dyna_nelth;
  size_t iElement = 0;
  size_t iData = 0;
  // Loop over elements
  for (int32_t ii = wordPosition; ii < wordPosition + wordsToRead;
       ii += nVarsElem4th) {
    // Loop over element data
    iData = 0;
    for (int32_t jj = ii; jj < ii + nVarsElem4th; jj++) {
      buffer_elems4th[iElement][iData++] = buffer->read_int(jj);
    }

    ++iElement;
  }

  // Update word position
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  return std::move(buffer_elems4th);
}

/** Read the numbering of nodes and elements
 *
 * @return idvector : vector of (external) ids
 *
 * Read the numbering of the data into a 2d-vector.
 * numbering[iCategory][iIndex]
 * category: nodes  = 0
 *           solids = 1
 *           beams  = 2
 *           shells = 3
 *           tshells= 4
 */
std::vector<std::vector<int32_t>>
D3plot::read_geometry_numbering()
{
  // TODO
  // NARBS check wrong?!?!?!
  /*
  int32_t nHeaderWords = 10;
  int32_t kk = 10+dyna_numnp+dyna_nel2+dyna_nel4+dyna_nel8+dyna_nelth;
  if(dyna_narbs == kk){
    nHeaderWords = 10;
  } else if(dyna_narbs ==
  kk+dyna_nummat2+dyna_nummat4+dyna_nummat8+dyna_nummatth ){
    nHeaderWords = 16;
  } else {
    throw("Inconsistent definition of dyna_narbs detected.");
  }
  */

  if (this->dyna_narbs == 0)
    return std::vector<std::vector<int32_t>>();

#ifdef QD_DEBUG
  std::cout << "Reading mesh numbering at word " << wordPosition << " ... ";
#endif

  // pointer to nodes
  int32_t nsort = buffer->read_int(wordPosition);
  // pointer to elem8 numbering
  int32_t nsrh = buffer->read_int(wordPosition + 1);
  if (nsrh != dyna_numnp + abs(nsort)) {
#ifdef QD_DEBUG
    std::cout << "\n"
              << "iWord=" << wordPosition + 1 << "\n"
              << nsrh << " != " << dyna_numnp + abs(nsort) << "\n"
              << "dyna_numnp=" << dyna_numnp << "\n"
              << "abs(nsort)=" << abs(nsort) << std::endl;
    std::cout << "narbs=" << this->dyna_narbs << std::endl;
#endif
    throw(std::runtime_error(
      "nsrh != nsort + numnp is inconsistent in dyna file. Your "
      "file might be using FEMZIP."));
  }
  // pointer to elem2 numbering
  int32_t nsrb = buffer->read_int(wordPosition + 2);
  if (nsrb != nsrh + dyna_nel8)
    std::runtime_error(
      std::string("nsrb != nsrh + nel8 is inconsistent in dyna file."));
  // pointer to elem4 numbering
  int32_t nsrs = buffer->read_int(wordPosition + 3);
  if (nsrs != nsrb + dyna_nel2)
    throw(
      std::runtime_error("nsrs != nsrb + nel2 is inconsistent in dyna file."));
  // pointer to elemth numbering
  int32_t nsrt = buffer->read_int(wordPosition + 4);
  if (nsrt != nsrs + dyna_nel4)
    throw(
      std::runtime_error("nsrt != nsrs + nel4 is inconsistent in dyna file."));
  // nNode consistent?
  if (buffer->read_int(wordPosition + 5) != dyna_numnp)
    throw(std::runtime_error(
      "Number of nodes is not defined consistently in d3plot geometry "
      "section."));

  /* === ID - ORDER === */
  // nodes,solids,beams,shells,tshells

  std::vector<std::vector<int32_t>> idvector(5);

  // Node IDs
  if (nsort < 0) {
    wordPosition += 16;
  } else {
    wordPosition += 10;
  }
  // wordPosition += 16; // header length is 16
  wordsToRead = dyna_numnp;
  std::vector<int32_t> nodeIDs(wordsToRead);
  size_t jj = 0;
  for (int32_t ii = wordPosition; ii < wordPosition + wordsToRead; ++ii) {
    nodeIDs[jj++] = buffer->read_int(ii);
  }
  idvector[0] = nodeIDs;

  // Solid IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nel8;
  std::vector<int32_t> solidIDs(wordsToRead);
  jj = 0;
  for (int32_t ii = wordPosition; ii < wordPosition + wordsToRead; ++ii) {
    solidIDs[jj++] = buffer->read_int(ii);
  }
  idvector[1] = solidIDs;

  // Beam IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nel2;
  std::vector<int32_t> beamIDs(wordsToRead);
  jj = 0;
  for (int32_t ii = wordPosition; ii < wordPosition + wordsToRead; ++ii) {
    beamIDs[jj++] = buffer->read_int(ii);
  }
  idvector[2] = beamIDs;

  // Shell IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nel4;
  std::vector<int32_t> shellIDs(wordsToRead);
  jj = 0;
  for (int32_t ii = wordPosition; ii < wordPosition + wordsToRead; ++ii) {
    shellIDs[jj++] = buffer->read_int(ii);
  }
  idvector[3] = shellIDs;

  // Thick Shell IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nelth;
  std::vector<int32_t> thick_shell_ids(wordsToRead);
  jj = 0;
  for (int32_t ii = wordPosition; ii < wordPosition + wordsToRead; ++ii) {
    thick_shell_ids[jj++] = buffer->read_int(ii);
  }
  idvector[4] = thick_shell_ids;
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  return std::move(idvector);
}

/** Read the numbering of the parts
 *
 * @return part_numbering : numbering of the parts
 */
std::vector<int32_t>
D3plot::read_part_ids()
{

  /*
   * Indeed this is a little complicated: usually the file should contain
   * as many materials as in the input but somehow dyna generates a few
   * ghost materials itself and those are appended with a 0 ID. Therefore
   * the length should be nMaterials but it's d3plot_nmmat with:
   * nMaterials < d3plot_nmmat. The difference are the ghost mats.
   * Took some time to find that out ... and I don't know why ...
   * oh and it is undocumented ...
   */

#ifdef QD_DEBUG
  std::cout << "Reading part numbering at word " << wordPosition << " ... ";
#endif

  std::vector<int32_t> part_numbering(dyna_nmmat);

  // std::vector<int32_t> externalPartIDs(nMaterials);
  wordsToRead = 3 * dyna_nmmat;

  int32_t jj = 0;
  for (int32_t ii = wordPosition; ii < wordPosition + dyna_nmmat; ++ii) {
    part_numbering[jj++] = buffer->read_int(ii);
  }

  /* sorted ids and sort indices are not needed
  jj = 0;
  for (int32_t ii = wordPosition + dyna_nmmat;
       ii < wordPosition + 2 * dyna_nmmat;
       ++ii) {
    part_numbering[1][jj++] = buffer->read_int(ii);
  }
  std::cout << std::endl;

  jj = 0;
  for (int32_t ii = wordPosition + 2 * dyna_nmmat;
       ii < wordPosition + 3 * dyna_nmmat;
       ++ii) {
    part_numbering[2][jj++] = buffer->read_int(ii);
  }
  std::cout << std::endl;
  */

  // update position
  // wordPosition += dyna_narbs;
  /*
   * the whole numbering section should have length narbs
   * but the procedure here should do the same ... hopefully
   */
  wordPosition += wordsToRead;

  // extra node elements skipped
  // 10 node solids: 2 node conn
  if (own_nel10)
    wordPosition += 2 * abs(dyna_nel8);
  // 8 node shells: 4 node conn
  if (dyna_nel48 > 0)
    wordPosition += 5 * dyna_nel48;
  // 20 node solids: 12 node conn
  if ((dyna_extra > 0) && (dyna_nel20 > 0))
    wordPosition += 13 * dyna_nel20;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  return part_numbering;
}

/** Read the geometry data for the airbags
 *
 */
void
D3plot::read_geometry_airbag()
{

  if (this->dyna_npefg > 0) {

    // ?!?!?!?!
    if (this->dyna_npefg / 10000000 == 1) {
      int32_t dyna_airbag_des = this->dyna_npefg / 10000000;
      wordPosition += dyna_airbag_des;
    }

    // skip airbag infos
    if (this->dyna_airbag_ngeom == 5) {
      wordPosition += 5 * this->dyna_airbag_npartgas;
    } else {
      wordPosition += 4 * this->dyna_airbag_npartgas;
    }

    /*
for (auto iAirbag = 0; iAirbag < this->dyna_airbag_npartgas; ++iAirbag) {
  // 1. particle id
  ++wordPosition;
  // 2. number of particles
  ++wordPosition;
  // 3. id for airbag
  ++wordPosition;
  // 4. number of gas mixtures
  ++wordPosition;
  if (this->dyna_airbag_ngeom == 5) {
    // 5. nchambers
    ++wordPosition;
  }
}
*/
  } // if dyna_npefg
}

/** Read the part names from the geometry section
 *
 * @param _part_ids : array of exernal part ids
 */
std::vector<std::string>
D3plot::read_part_names()
{
#ifdef QD_DEBUG
  std::cout << "Reading part info ... ";
#endif

  int32_t ntype = this->buffer->read_int(wordPosition);
  if (ntype != 90001) {
    throw(std::runtime_error("ntype must be 90001 in part section."));
  }

  this->dyna_numprop = this->buffer->read_int(wordPosition + 1);
  if (this->dyna_numprop < 0)
    throw(std::runtime_error(
      "negative number of parts in part section makes no sense."));

  std::vector<std::string> part_names;
  part_names.reserve(this->dyna_numprop);
  for (int32_t ii = 0; ii < this->dyna_numprop; ii++) {

    // start of the section of current part in the file
    int32_t start = (wordPosition + 1) + ii * 19 + 1;

    // this id is wrong ... and don't ask me why
    // int32_t partID = this->buffer->read_int(start);
    part_names.push_back(this->buffer->read_str(start + 1, 18));

    // createpart
    // this->get_db_parts()->add_partByID(part_id)->set_name(partName);
  }

#ifdef QD_DEBUG
  std::cout << this->get_db_parts()->get_nParts() << " done." << std::endl;
#endif

  // update position
  wordPosition += 1 + (this->dyna_numprop + 1) * 19 + 1;

  return part_names;
}

/*
 * Check for the file ending marker. That is a float
 * of -999999.
 */
bool
D3plot::isFileEnding(int32_t iWord)
{
#ifdef QD_DEBUG
  std::cout << "Checking end at word " << iWord << " ... ";
#endif

  if (this->buffer->read_float(iWord) + 999999.0f == 0.) {
#ifdef QD_DEBUG
    std::cout << "ok (true)" << std::endl;
#endif
    return true;
  } else {
#ifdef QD_DEBUG
    std::cout << "ok (false)" << std::endl;
#endif
    return false;
  }
}

/*
 * Parse the user input for state reading.
 *
 */
void
D3plot::read_states_parse(std::vector<std::string> _variables)
{
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

  for (size_t ii = 0; ii < _variables.size(); ++ii) {
    // Displacement
    if (_variables[ii].find("disp") != std::string::npos) {
      if (dyna_iu == 0)
        throw(std::invalid_argument(
          "Unable to read displacements, since there are none."));
      this->disp_read = read_states_parse_readMode(_variables[ii]);

      if (this->disp_is_read) {
#ifdef QD_DEBUG
        std::cout << "disp already loaded." << std::endl;
#endif
        this->disp_read = 0;
      }
      // Velocity
    } else if (_variables[ii].find("vel") != std::string::npos) {
      if (dyna_iv == 0)
        throw(std::invalid_argument(
          "Unable to read velocities, since there are none."));
      this->vel_read = read_states_parse_readMode(_variables[ii]);

      if (this->vel_is_read) {
#ifdef QD_DEBUG
        std::cout << "vel already loaded." << std::endl;
#endif
        this->vel_read = 0;
      }
      // Acceleration
    } else if (_variables[ii].find("accel") != std::string::npos) {
      if (dyna_ia == 0)
        throw(std::invalid_argument(
          "Unable to read accelerations, since there are none."));
      this->acc_read = read_states_parse_readMode(_variables[ii]);

      if (this->acc_is_read) {
#ifdef QD_DEBUG
        std::cout << "accel already loaded." << std::endl;
#endif
        this->acc_read = 0;
      }
      // Mises Stress ( must be before stress! )
    } else if (_variables[ii].find("stress_mises") != std::string::npos) {
#ifdef QD_DEBUG
      if (dyna_ioshl1 == 0)
        std::cout << "Warning: There are no shell-stresses in the file."
                  << std::endl;
#endif
      this->stress_mises_read = read_states_parse_readMode(_variables[ii]);

      if (this->stress_mises_is_read) {
#ifdef QD_DEBUG
        std::cout << "stress_mises already loaded." << std::endl;
#endif
        this->stress_mises_read = 0;
      }

      // Stress
    } else if (_variables[ii].find("stress") != std::string::npos) {
#ifdef QD_DEBUG
      if (dyna_ioshl1 == 0)
        std::cout << "Warning: There are no shell-stresses in the file."
                  << std::endl;
#endif
      this->stress_read = read_states_parse_readMode(_variables[ii]);

      if (this->stress_is_read) {
#ifdef QD_DEBUG
        std::cout << "stress already loaded." << std::endl;
#endif
        this->stress_read = 0;
      }
      // Plastic Strain
      // must be before strain !!!!
    } else if (_variables[ii].find("plastic_strain") != std::string::npos) {
#ifdef QD_DEBUG
      if (dyna_ioshl2 == 0)
        std::cout << "Warning: There are no shell-plastic-strains in the file."
                  << std::endl;
#endif
      this->plastic_strain_read = read_states_parse_readMode(_variables[ii]);

      if (this->plastic_strain_is_read) {
#ifdef QD_DEBUG
        std::cout << "plastic strain already loaded." << std::endl;
#endif
        this->plastic_strain_read = 0;
      }
      // Strain
    } else if (_variables[ii].find("strain") != std::string::npos) {
      if (dyna_istrn == 0)
        throw(std::invalid_argument(
          "Unable to read strains, since there are none."));
      this->strain_read = read_states_parse_readMode(_variables[ii]);

      if (this->strain_is_read) {
#ifdef QD_DEBUG
        std::cout << "strain already loaded." << std::endl;
#endif
        this->strain_read = 0;
      }
      // Internal Energy
    } else if (_variables[ii].find("energy") != std::string::npos) {
      if (dyna_ioshl4 == 0)
        throw(std::invalid_argument(
          "Unable to read energies, since there are none."));
      this->energy_read = read_states_parse_readMode(_variables[ii]);

      if (this->energy_is_read) {
#ifdef QD_DEBUG
        std::cout << "energy already loaded." << std::endl;
#endif
        this->energy_read = 0;
      }
      // History variables
    } else if (_variables[ii].find("history") != std::string::npos) {
      // retrieve history var indexesindexes
      auto hist_vars = extract_integers<int32_t>(_variables[ii]);
      if (hist_vars.size() < 1)
        throw(std::invalid_argument(
          "No history variable index specified. Please input at "
          "least one number seperated by spaces."));
      auto var_mode = read_states_parse_readMode(_variables[ii]);

      /* SHELLS */
      if (_variables[ii].find("shell") != std::string::npos) {
        // Check: already loaded
        for (size_t jj = 0; jj < this->history_shell_is_read.size(); ++jj) {
          auto kk = find(hist_vars.begin(),
                         hist_vars.end(),
                         this->history_shell_is_read[jj]);
          if (kk != hist_vars.end()) {
#ifdef QD_DEBUG
            std::cout << "history variable " << *kk
                      << " already loaded for shells." << std::endl;
#endif
            hist_vars.erase(kk);
          }
        }

        // Check: already defined in an argument previously
        for (size_t jj = 0; jj < this->history_shell_read.size(); ++jj) {
          auto kk = find(
            hist_vars.begin(), hist_vars.end(), this->history_shell_read[jj]);

          if (kk != hist_vars.end()) {
            std::cout << "Warning: trying to read history variable " << *kk
                      << " twice for shells, using only first occurrence."
                      << std::endl;
            hist_vars.erase(kk);
          }
        }

        // Check: var index beyond limit neips
        for (size_t jj = 0; jj < hist_vars.size(); ++jj) {
          if (hist_vars[jj] < 1) {
            throw(std::invalid_argument(
              "History variable index must be at least 1 and not " +
              std::to_string(hist_vars[jj])));
          }
          if (hist_vars[jj] > this->dyna_neips) {
            throw(std::invalid_argument("Warning: history variable " +
                                        std::to_string(hist_vars[jj]) +
                                        " exceeds the limit for shells of " +
                                        std::to_string(this->dyna_neips)));
          }
        }

        // Save var indexes and var modes
        for (size_t jj = 0; jj < hist_vars.size(); ++jj) {
          this->history_shell_read.push_back(hist_vars[jj]);
          this->history_shell_mode.push_back(var_mode);
        }

        /* SOLIDS */
      } else if (_variables[ii].find("solid") != std::string::npos) {
        // Check: already loaded
        for (size_t jj = 0; jj < this->history_solid_is_read.size(); ++jj) {
          auto kk = find(hist_vars.begin(),
                         hist_vars.end(),
                         this->history_solid_is_read[jj]);
          if (kk != hist_vars.end()) {
#ifdef QD_DEBUG
            std::cout << "history variable " << *kk
                      << " already loaded for solids." << std::endl;
#endif
            hist_vars.erase(kk);
          }
        }

        // Check: already defined in an argument previously
        for (size_t jj = 0; jj < this->history_solid_read.size(); ++jj) {
          auto kk = find(
            hist_vars.begin(), hist_vars.end(), this->history_solid_read[jj]);

          if (kk != hist_vars.end()) {
            std::cout << "Warning: trying to read history variable " << *kk
                      << " twice for solids, using only first occurrence."
                      << std::endl;
            hist_vars.erase(kk);
          }
        }

        // Check: var index beyond limit neiph
        for (size_t jj = 0; jj < hist_vars.size(); ++jj) {

          if (hist_vars[jj] < 1) {
            throw(std::invalid_argument(
              "History variable index must be at least 1."));
          }

          if (hist_vars[jj] > this->dyna_neiph) {
            throw(std::invalid_argument("Warning: history variable " +
                                        std::to_string(hist_vars[jj]) +
                                        " exceeds the limit for solids of " +
                                        std::to_string(this->dyna_neiph)));
          }
        }

        // save var indexes
        for (size_t jj = 0; jj < hist_vars.size(); ++jj) {
          this->history_solid_read.push_back(hist_vars[jj]);
          this->history_solid_mode.push_back(var_mode);
        }

        // unknown element type
      } else {
        throw(std::invalid_argument(
          "Please specify the element type for all history "
          "variables as either shell or solid"));
      }

    } else {
      throw(std::invalid_argument("Reading of variable:" + _variables[ii] +
                                  " is undefined"));
    } // if:variable.find
  }   // for:variables
}

/*
 * Returns the int32_t code for the read mode of the state variables in the
 * d3plot.
 * Modes are: min,max,outer,mid,inner,mean
 */
int32_t
D3plot::read_states_parse_readMode(const std::string& _variable) const
{
  if (_variable.find("max") != std::string::npos) {
    return 1;
  } else if (_variable.find("min") != std::string::npos) {
    return 2;
  } else if (_variable.find("outer") != std::string::npos) {
    return 3;
  } else if (_variable.find("mid") != std::string::npos) {
    return 4;
  } else if (_variable.find("inner") != std::string::npos) {
    return 5;
  } else if (_variable.find("mean") != std::string::npos) {
    return 6;
  } else {
    return 6; // std is mean
  }
}

/** Read the state data.
 *
 * @param _variable variable to load
 */
void
D3plot::read_states(const std::string& _variable)
{
  std::vector<std::string> vec = { _variable };
  this->read_states(vec);
};

/** Read the state data.
 *
 * @param _variables std::vector of variables to read
 */
void
D3plot::read_states(std::vector<std::string> _variables)
{
#ifdef QD_DEBUG
  std::cout << "> STATES" << std::endl;
  for (size_t ii = 0; ii < _variables.size(); ++ii)
    std::cout << "variable: " << _variables[ii] << std::endl;
#endif

  if ((_variables.size() < 1) && (this->timesteps.size() > 0))
    throw(
      std::invalid_argument("The list of state variables to load is empty."));

  // Decode variable reading
  this->read_states_parse(_variables);

  // Check if variables are already read.
  // If just 1 is not read yet, the routine runs through.
  // Will not work first time :P since we must run through at least once
  if ((this->disp_read + this->vel_read + this->acc_read +
         this->plastic_strain_read + this->energy_read + this->strain_read +
         this->stress_read + this->stress_mises_read +
         this->history_shell_read.size() + this->history_solid_read.size() ==
       0) &&
      (this->timesteps.size() != 0))
    return;

  // Calculate loop properties
  size_t iState = 0;
  int32_t nVarsNodes =
    (dyna_ndim * (dyna_iu + dyna_iv + dyna_ia) + own_has_mass_scaling_info) *
    dyna_numnp;
  int32_t nVarsElems = dyna_nel2 * dyna_nv1d +
                       (dyna_nel4 - dyna_numrbe) * dyna_nv2d +
                       dyna_nel8 * dyna_nv3d + dyna_nelth * dyna_nv3dt;
  int32_t nAirbagVars =
    this->dyna_airbag_npartgas * this->dyna_airbag_state_geom +
    this->dyna_airbag_nparticles * this->dyna_airbag_state_nvars;

  // Variable Deletion table
  int32_t nDeletionVars = 0;
  if (dyna_mdlopt == 0) {
    // ok
  } else if (dyna_mdlopt == 1) {
    nDeletionVars = dyna_numnp;
  } else if (dyna_mdlopt == 2) {
    nDeletionVars = dyna_nel2 + dyna_nel4 + dyna_nel8 + dyna_nelth;
  } else {
    throw(std::runtime_error("Parameter mdlopt:" + std::to_string(dyna_mdlopt) +
                             " makes no sense."));
  }

  // Checks for timesteps
  bool timesteps_read = false;
  if (this->timesteps.size() < 1)
    timesteps_read = true;

  bool firstFileDone = false;

  // Check for first time
  // Makes no difference for D3plotBuffer but for
  // the FemzipBuffer.
  if (this->timesteps.size() < 1) {
    this->buffer->init_nextState();
    this->wordPositionStates = this->wordPosition;
  } else {
    this->buffer->rewind_nextState();
    this->wordPosition = this->wordPositionStates;
  }

  // Loop over state files
  while (this->buffer->has_nextState()) {
    this->buffer->read_nextState();

    // Not femzip case
    if ((!this->_is_femzipped) && firstFileDone) {
      wordPosition = 0;
    }
    // femzip case
    if (this->_is_femzipped) {
      // 0 = endmark
      // 1 = ntype = 90001
      // 2 = numprop
      int32_t dyna_numprop_states = this->buffer->read_int(2);
      if (this->dyna_numprop != dyna_numprop_states)
        throw(std::runtime_error(
          "Numprop in geometry section != numprop in states section!"));
      wordPosition = 1; // endline symbol at 0 in case of femzip ...
      wordPosition += 1 + (this->dyna_numprop + 1) * 19 + 1;
      // this->femzip_state_offset = wordPosition;
    }

    // Loop through states
    while (!this->isFileEnding(wordPosition)) {

      if (timesteps_read) {
        float state_time = buffer->read_float(wordPosition);
        this->timesteps.push_back(state_time);
#ifdef QD_DEBUG
        std::cout << "State: " << iState << " Time: " << state_time
                  << " at word " << wordPosition << std::endl;
#endif
      }

      // NODE - DISP
      if (dyna_iu && (this->disp_read != 0)) {
        read_states_displacement();
      }

      // NODE - VEL
      if (dyna_iv && (this->vel_read != 0)) {
        read_states_velocity();
      }

      // NODE - ACCEL
      if (dyna_ia && (this->acc_read != 0)) {
        read_states_acceleration();
      }

      // ELEMENT - STRESS, STRAIN, ENERGY, PLASTIC STRAIN
      if (this->stress_read || this->stress_mises_read || this->strain_read ||
          this->energy_read || this->plastic_strain_read ||
          this->history_shell_read.size() || this->history_solid_read.size()) {

        // solids
        read_states_elem8(iState);
        // thick shells
        read_states_elem4th(iState);
        // shells
        read_states_elem4(iState);
      }

      // read_states_airbag(); // skips airbag section

      // update position
      // +1 is just for time word
      wordPosition +=
        nAirbagVars + nVarsNodes + nVarsElems + nDeletionVars + dyna_nglbv + 1;

      iState++;
    }

    firstFileDone = true;
  }

  this->buffer->end_nextState();

  // Set, which variables were read
  if (this->disp_read != 0) {
    this->disp_is_read = true;
  }
  if (this->vel_read != 0) {
    this->vel_is_read = true;
  }
  if (this->plastic_strain_read != 0) {
    this->plastic_strain_is_read = true;
  }
  if (this->energy_read != 0) {
    this->energy_is_read = true;
  }
  if (this->strain_read != 0) {
    this->strain_is_read = true;
  }
  if (this->stress_read != 0) {
    this->stress_is_read = true;
  }
  if (this->stress_mises_read != 0) {
    this->stress_mises_is_read = true;
  }
  for (size_t ii = 0; ii < this->history_shell_read.size(); ++ii) {
    this->history_shell_is_read.push_back(this->history_shell_read[ii]);
  }
  for (size_t ii = 0; ii < this->history_solid_read.size(); ++ii) {
    this->history_solid_is_read.push_back(this->history_solid_read[ii]);
  }
}

/*
 * Read the node displacement into the db.
 *
 */
void
D3plot::read_states_displacement()
{
  if (dyna_iu != 1)
    return;

  int32_t start = wordPosition + dyna_nglbv + 1;

  wordsToRead = dyna_numnp * dyna_ndim;
  // size_t iNode = 0;

#ifdef QD_DEBUG
  std::cout << "> read_states_displacement at " << start << std::endl;
#endif

  DB_Nodes* db_nodes = this->get_db_nodes();
  const auto nNodes = static_cast<int32_t>(db_nodes->get_nNodes());

#pragma omp parallel
  {
    std::vector<float> disp_tmp(dyna_ndim);

#pragma omp for schedule(dynamic)
    for (int32_t iNode = 0; iNode < nNodes; ++iNode) {
      auto ii = start + iNode * dyna_ndim;
      buffer->read_float_array(ii, dyna_ndim, disp_tmp);
      db_nodes->get_nodeByIndex(iNode)->add_disp(disp_tmp);
    }
  }

  // old
  // for (int32_t ii = start; ii < start + wordsToRead; ii += dyna_ndim) {
  //   auto node = db_nodes->get_nodeByIndex(iNode);

  //   buffer->read_float_array(ii, dyna_ndim, _disp);
  //   node->add_disp(_disp);

  //   ++iNode;
  // }
}

/*
 * Read the node velocity.
 *
 */
void
D3plot::read_states_velocity()
{
  if (dyna_iv != 1)
    return;

  int32_t start =
    wordPosition + 1 + dyna_nglbv +
    (dyna_iu * dyna_ndim + own_has_mass_scaling_info) * dyna_numnp;

  wordsToRead = dyna_numnp * dyna_ndim;

#ifdef QD_DEBUG
  std::cout << "> read_states_velocity at " << start << std::endl;
#endif

  DB_Nodes* db_nodes = this->get_db_nodes();
  const auto nNodes = static_cast<int64_t>(db_nodes->get_nNodes());

#pragma omp parallel
  {
    std::vector<float> vel_tmp(dyna_ndim);

#pragma omp for schedule(dynamic)
    for (int32_t iNode = 0; iNode < nNodes; ++iNode) {
      auto ii = start + iNode * dyna_ndim;
      buffer->read_float_array(ii, dyna_ndim, vel_tmp);
      db_nodes->get_nodeByIndex(iNode)->add_vel(vel_tmp);
    }
  }
}

/*
 * Read the node acceleration.
 *
 */
void
D3plot::read_states_acceleration()
{
  if (dyna_ia != 1)
    return;

  int32_t start =
    wordPosition + 1 + dyna_nglbv +
    ((dyna_iu + dyna_iv) * dyna_ndim + own_has_mass_scaling_info) * dyna_numnp;

  wordsToRead = dyna_numnp * dyna_ndim;

#ifdef QD_DEBUG
  std::cout << "> read_states_acceleration at " << start << std::endl;
#endif

  DB_Nodes* db_nodes = this->get_db_nodes();
  const auto nNodes = static_cast<int64_t>(db_nodes->get_nNodes());

#pragma omp parallel
  {
    std::vector<float> accel_tmp(dyna_ndim);

#pragma omp for schedule(dynamic)
    for (int32_t iNode = 0; iNode < nNodes; ++iNode) {
      auto ii = start + iNode * dyna_ndim;
      buffer->read_float_array(ii, dyna_ndim, accel_tmp);
      db_nodes->get_nodeByIndex(iNode)->add_accel(accel_tmp);
    }
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
void
D3plot::read_states_elem8(size_t iState)
{
  if ((dyna_nv3d <= 0) && (dyna_nel8 <= 0))
    return;

  int32_t start =
    this->wordPosition + 1 // time
    + dyna_nglbv +
    ((dyna_iu + dyna_iv + dyna_ia) * dyna_ndim + own_has_mass_scaling_info) *
      dyna_numnp;

  wordsToRead = dyna_nv3d * dyna_nel8;

  std::vector<float> tmp_vector(6);
  std::vector<float> history_vars(this->history_solid_read.size());
  DB_Elements* db_elements = this->get_db_elements();

  size_t iElement = 0;
  for (int32_t ii = start; ii < start + wordsToRead; ii += dyna_nv3d) {
    auto element = db_elements->get_elementByIndex(Element::SOLID, iElement);

    // stress tensor and data
    if (this->stress_read || this->stress_mises_read) {
      // tmp_vector.clear();
      buffer->read_float_array(ii, 6, tmp_vector);

      if (this->stress_read)
        element->add_stress(tmp_vector);
      if (this->stress_mises_read)
        element->add_stress_mises(MathUtility::mises_stress(tmp_vector));
    }

    // plastic strain
    if (this->plastic_strain_read) {
      element->add_plastic_strain(this->buffer->read_float(ii + 6));
    }

    // strain tensor
    if ((dyna_istrn == 1) && this->strain_read) {
      // tmp_vector.clear();
      buffer->read_float_array(ii + dyna_nv3d - 6, 6, tmp_vector);
      element->add_strain(tmp_vector);
    }

    // no energy ...

    // history variables
    if (history_solid_read.size()) {
      history_vars.clear();

      for (size_t jj = 0; jj < history_solid_read.size(); ++jj) {

        // Skip if over limit
        if (this->history_solid_read[jj] > this->dyna_neiph)
          continue;

        history_vars.push_back(
          this->buffer->read_float(ii + 6 + history_solid_read[jj]));

      } // loop:history
      element->add_history_vars(history_vars, iState);
    } // if:history

    iElement++;
  }
}

/* Read the state data of the shell elements
 *
 * > strain tensor
 * > strain mises
 * > stress tensor
 * > stress mises
 * > eq. plastic strain
 * > internal energy
 */
void
D3plot::read_states_elem4(size_t iState)
{
  if ((dyna_istrn != 1) && (dyna_nv2d <= 0) && (dyna_nel4 - dyna_numrbe > 0))
    return;

  // prepare looping
  const int32_t start =
    this->wordPosition + 1 // time
    + dyna_nglbv +
    ((dyna_iu + dyna_iv + dyna_ia) * dyna_ndim + own_has_mass_scaling_info) *
      dyna_numnp +
    dyna_nv3d * dyna_nel8 + dyna_nelth * dyna_nv3dt + dyna_nv1d * dyna_nel2;

  wordsToRead = dyna_nv2d * (dyna_nel4 - dyna_numrbe);

  // offsets
  const int32_t iPlastStrainOffset = this->dyna_ioshl1 * 6; // stresses before?
  const int32_t iHistoryOffset =
    iPlastStrainOffset + this->dyna_ioshl2; // stresses & pl. strain before
  const int32_t iLayerSize = dyna_neips + iHistoryOffset;

  // const auto nElements_shell = static_cast<int64_t>(
  //   get_db_elements()->get_nElements(Element::ElementType::SHELL));

  // #pragma omp parallel
  {

    // helpful vars
    // vectors
    std::vector<float> tmp_vec6(6);
    std::vector<float> layers_stress_mises(dyna_maxint);
    std::vector<float> layers_plastic_strain(dyna_maxint);
    // matrices
    std::vector<std::vector<float>> layers_stress(
      6, std::vector<float>(dyna_maxint));
    std::vector<std::vector<float>> layers_strain(6, std::vector<float>(2));
    std::vector<std::vector<float>> layers_history(
      this->history_shell_read.size(), std::vector<float>(dyna_maxint));

    // Do the thing ...
    size_t iElement = 0;
    for (int32_t ii = start; ii < start + wordsToRead; ++iElement) {

      // #pragma omp for schedule(dynamic)
      // #pragma omp for
      // for (int32_t iElement = 0; iElement < nElements_shell; ++iElement) {

      // auto ii = start + element_offsets[iElement];
      // const auto ii = start + iElement * dyna_nv2d;

      // get element (and check for rigidity)
      auto element =
        this->get_db_elements()->get_elementByIndex(Element::SHELL, iElement);

      // Fix:
      // Interestingly, dyna seems to write result values for rigid shells in
      // the d3part file, but not in the d3plot. Of course this is not
      // documented ...
      // 5 -> d3part
      if (dyna_filetype != 5 && element->get_is_rigid()) {
        // does not increment ii, but iElement!!!!!
        continue;
      }

      // LOOP: LAYERS
      for (int32_t iLayer = 0; iLayer < dyna_maxint; ++iLayer) {
        const int32_t layerStart = ii + iLayer * iLayerSize;

        // LAYER: STRESS TENSOR AND MISES
        if ((this->stress_read || this->stress_mises_read) && (dyna_ioshl1)) {
          layers_stress[0][iLayer] = this->buffer->read_float(layerStart);
          layers_stress[1][iLayer] = this->buffer->read_float(layerStart + 1);
          layers_stress[2][iLayer] = this->buffer->read_float(layerStart + 2);
          layers_stress[3][iLayer] = this->buffer->read_float(layerStart + 3);
          layers_stress[4][iLayer] = this->buffer->read_float(layerStart + 4);
          layers_stress[5][iLayer] = this->buffer->read_float(layerStart + 5);

          // stress mises calculation
          if (this->stress_mises_read) {
            tmp_vec6[0] = layers_stress[0][iLayer];
            tmp_vec6[1] = layers_stress[1][iLayer];
            tmp_vec6[2] = layers_stress[2][iLayer];
            tmp_vec6[3] = layers_stress[3][iLayer];
            tmp_vec6[4] = layers_stress[4][iLayer];
            tmp_vec6[5] = layers_stress[5][iLayer];
            layers_stress_mises[iLayer] = MathUtility::mises_stress(tmp_vec6);
          }

        } // end:stress

        // LAYER: PLASTIC_STRAIN
        if ((this->plastic_strain_read) && (dyna_ioshl2)) {

          layers_plastic_strain[iLayer] =
            this->buffer->read_float(layerStart + iPlastStrainOffset);
        }

        // LAYERS: HISTORY SHELL
        if (this->dyna_neips) {
          int32_t iHistoryStart = layerStart + iHistoryOffset - 1;

          for (size_t iHistoryVar = 0;
               iHistoryVar < this->history_shell_read.size();
               ++iHistoryVar) {

            // history vars start with index 1 and not 0, thus the -1
            layers_history[iHistoryVar][iLayer] = this->buffer->read_float(
              iHistoryStart + history_shell_read[iHistoryVar]);

          } // loop:history
        }   // if:history

      } // loop:layers

      // add layer vars (if requested)
      if (dyna_ioshl2 && this->plastic_strain_read)
        element->add_plastic_strain(compute_state_var_from_mode(
          layers_plastic_strain, this->plastic_strain_read));
      if (dyna_ioshl1 && this->stress_read)
        element->add_stress(
          compute_state_var_from_mode(layers_stress, this->stress_read));
      if (dyna_ioshl1 && this->stress_mises_read)
        element->add_stress_mises(compute_state_var_from_mode(
          layers_stress_mises, this->stress_mises_read));
      if (this->history_shell_read.size())
        element->add_history_vars(
          compute_state_var_from_mode(layers_history, this->history_shell_mode),
          iState);

      // STRAIN TENSOR
      if (dyna_istrn && this->strain_read) {

        int32_t strainStart = this->own_has_internal_energy
                                ? ii + dyna_nv2d - 13
                                : ii + dyna_nv2d - 12;

        layers_strain[0][0] = this->buffer->read_float(strainStart);
        layers_strain[1][0] = this->buffer->read_float(strainStart + 1);
        layers_strain[2][0] = this->buffer->read_float(strainStart + 2);
        layers_strain[3][0] = this->buffer->read_float(strainStart + 3);
        layers_strain[4][0] = this->buffer->read_float(strainStart + 4);
        layers_strain[5][0] = this->buffer->read_float(strainStart + 5);
        layers_strain[0][1] = this->buffer->read_float(strainStart + 6);
        layers_strain[1][1] = this->buffer->read_float(strainStart + 7);
        layers_strain[2][1] = this->buffer->read_float(strainStart + 8);
        layers_strain[3][1] = this->buffer->read_float(strainStart + 9);
        layers_strain[4][1] = this->buffer->read_float(strainStart + 10);
        layers_strain[5][1] = this->buffer->read_float(strainStart + 11);

        element->add_strain(
          compute_state_var_from_mode(layers_strain, this->strain_read));
      }

      // INTERNAL ENERGY
      if (this->energy_read && this->own_has_internal_energy) {
        element->add_energy(this->buffer->read_float(ii + dyna_nv2d - 1));
      }

      /*
      // internal energy (a little complicated ... but that's dyna)
      if (this->energy_read && dyna_ioshl4) {
        if (dyna_istrn == 1) {
          if (dyna_nv2d >= 45)
            element->add_energy(this->buffer->read_float(ii + dyna_nv2d -
      1)); } else { element->add_energy(this->buffer->read_float(ii +
      dyna_nv2d - 1));
        }
      }
      */

      ii += dyna_nv2d;
    } // for elements
  }   // pragma omp parallel
}

/** Read the state data of the thick shell elements
 *
 * @param iState : current state
 */
void
D3plot::read_states_elem4th(size_t iState)
{

  if ((dyna_istrn != 1) && (dyna_nv3dt <= 0))
    return;

  // prepare looping
  int32_t start =
    this->wordPosition + 1 // time
    + dyna_nglbv +
    ((dyna_iu + dyna_iv + dyna_ia) * dyna_ndim + own_has_mass_scaling_info) *
      dyna_numnp +
    dyna_nv3d * dyna_nel8; // solids

  wordsToRead = dyna_nelth * dyna_nv3dt;

  // offsets
  int32_t iPlastStrainOffset = this->dyna_ioshl1 * 6; // stresses before?
  int32_t iHistoryOffset =
    iPlastStrainOffset + this->dyna_ioshl2; // stresses & pl. strain before
  int32_t iLayerSize = dyna_neips + iHistoryOffset;

  // helpful vars
  bool has_strains = this->dyna_neiph >= 6;
  // vectors
  std::vector<float> tmp_vec6(6);
  std::vector<float> layers_stress_mises(dyna_maxint);
  std::vector<float> layers_plastic_strain(dyna_maxint);
  // matrices
  std::vector<std::vector<float>> layers_stress(
    6, std::vector<float>(dyna_maxint));
  std::vector<std::vector<float>> layers_strain(6, std::vector<float>(2));
  std::vector<std::vector<float>> layers_history(
    this->history_shell_read.size(), std::vector<float>(dyna_maxint));

  // Do the thing ...
  size_t iElement = 0;
  for (int32_t ii = start; ii < start + wordsToRead; ++iElement) {

    // get element
    auto element =
      this->get_db_elements()->get_elementByIndex(Element::TSHELL, iElement);

    // LOOP: LAYERS
    for (int32_t iLayer = 0; iLayer < dyna_maxint; ++iLayer) {
      int32_t layerStart = ii + iLayer * iLayerSize;

      // LAYER: PLASTIC_STRAIN
      if ((this->plastic_strain_read) && (dyna_ioshl2)) {

        layers_plastic_strain[iLayer] =
          this->buffer->read_float(layerStart + iPlastStrainOffset);
      }

      // LAYER: STRESS TENSOR AND MISES
      if ((this->stress_read || this->stress_mises_read) &&
          (this->dyna_ioshl1)) {

        layers_stress[0][iLayer] = this->buffer->read_float(layerStart);
        layers_stress[1][iLayer] = this->buffer->read_float(layerStart + 1);
        layers_stress[2][iLayer] = this->buffer->read_float(layerStart + 2);
        layers_stress[3][iLayer] = this->buffer->read_float(layerStart + 3);
        layers_stress[4][iLayer] = this->buffer->read_float(layerStart + 4);
        layers_stress[5][iLayer] = this->buffer->read_float(layerStart + 5);

        // stress mises calculation
        if (this->stress_mises_read) {
          tmp_vec6[0] = layers_stress[0][iLayer];
          tmp_vec6[1] = layers_stress[1][iLayer];
          tmp_vec6[2] = layers_stress[2][iLayer];
          tmp_vec6[3] = layers_stress[3][iLayer];
          tmp_vec6[4] = layers_stress[4][iLayer];
          tmp_vec6[5] = layers_stress[5][iLayer];
          layers_stress_mises[iLayer] = MathUtility::mises_stress(tmp_vec6);
        }

      } // end:stress

      // LAYERS: HISTORY SHELL
      if (this->dyna_neips) {
        int32_t iHistoryStart = layerStart + iHistoryOffset - 1;

        for (size_t iHistoryVar = 0;
             iHistoryVar < this->history_shell_read.size();
             ++iHistoryVar) {

          // history vars start with index 1 and not 0, thus the -1
          layers_history[iHistoryVar][iLayer] = this->buffer->read_float(
            iHistoryStart + history_shell_read[iHistoryVar]);

        } // loop:history
      }   // if:history

    } // loop:layers

    // add layer vars (if requested)
    if (dyna_ioshl2 && this->plastic_strain_read)
      element->add_plastic_strain(compute_state_var_from_mode(
        layers_plastic_strain, this->plastic_strain_read));
    if (dyna_ioshl1 && this->stress_read)
      element->add_stress(
        compute_state_var_from_mode(layers_stress, this->stress_read));
    if (dyna_ioshl1 && this->stress_mises_read)
      element->add_stress_mises(compute_state_var_from_mode(
        layers_stress_mises, this->plastic_strain_read));
    if (this->history_shell_read.size())
      element->add_history_vars(
        compute_state_var_from_mode(layers_history, this->history_shell_mode),
        iState);

    // STRAIN TENSOR
    if ((dyna_istrn == 1) && this->strain_read && has_strains) {

      int32_t strainStart =
        (dyna_nv2d >= 45) ? ii + dyna_nv2d - 13 : ii + dyna_nv2d - 12;

      layers_strain[0][0] = this->buffer->read_float(strainStart);
      layers_strain[1][0] = this->buffer->read_float(strainStart + 1);
      layers_strain[2][0] = this->buffer->read_float(strainStart + 2);
      layers_strain[3][0] = this->buffer->read_float(strainStart + 3);
      layers_strain[4][0] = this->buffer->read_float(strainStart + 4);
      layers_strain[5][0] = this->buffer->read_float(strainStart + 5);
      layers_strain[0][1] = this->buffer->read_float(strainStart + 6);
      layers_strain[1][1] = this->buffer->read_float(strainStart + 7);
      layers_strain[2][1] = this->buffer->read_float(strainStart + 8);
      layers_strain[3][1] = this->buffer->read_float(strainStart + 9);
      layers_strain[4][1] = this->buffer->read_float(strainStart + 10);
      layers_strain[5][1] = this->buffer->read_float(strainStart + 11);

      element->add_strain(
        compute_state_var_from_mode(layers_strain, this->strain_read));
    }

    // no internal energy for tshells?

    ii += dyna_nv3dt;
  } // for elements
}

/** Read the airbag state data
 *
 * Airbag data and particle data (skipped currently)
 */
void
D3plot::read_states_airbag()
{

  /*
  int32_t start = this->wordPosition + 1 // time
                  + dyna_nglbv +
                  (dyna_iu + dyna_iv + dyna_ia) * dyna_numnp * dyna_ndim +
                  dyna_nv3d * dyna_nel8 + dyna_nelth * dyna_nv3dt +
                  dyna_nv1d * dyna_nel2 + dyna_nv2d * dyna_nel4;
  */

  // Airbag geometry data
  // wordsToRead = this->dyna_airbag_npartgas * this->dyna_airbag_state_geom;
  // for dyna_airbag_npartgas
  // 1. number of active particles
  // 2. current bag volume

  // Particle state data
  // wordsToRead = dyna_airbag_nparticles * dyna_airbag_state_nvars;
  // for dyna_airbag_nparticles
  // 1. gas id
  // 2. chamber id
  // 3. leakage (0 active, -1 fabric, -2 vent hole, -3 mistracked)
  // 4. mass
  // 5. radius
  // 6. spin energy
  // 7. translation energy
  // 8. distance from particle to nearest segmen
  // 9. x position
  // 10. y position
  // 11. z position
  // 12. x velocity
  // 13. y velocity
  // 14. z velocity
}

/** Get the timestamps of the timesteps.
 *
 * @return timesteps : std::vector with the timestamp of the given state
 */
std::vector<float>
D3plot::get_timesteps() const
{
  return this->timesteps;
}

/** Get the title of the file in the header
 *
 * @return title
 */
std::string
D3plot::get_title() const
{
  return this->dyna_title;
}

/** Clears a loaded result
 *
 * @param _variable variable to clear
 *
 */
void
D3plot::clear(const std::string& _variable)
{
  std::vector<std::string> _variables = { _variable };
  this->clear(_variables);
};

/** Clears loaded result data loaded from the file
 *
 * @param _variables optional arg for cleansing only specific variables
 *
 * Clears all variables by default!
 */
void
D3plot::clear(const std::vector<std::string>& _variables)
{
  // Default: Clear all
  if (_variables.size() == 0) {
    // hihi this is a naughty trick ... calling myself again
    std::vector<std::string> _tmp;
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
    this->clear(_tmp);

  } else {
    // Convert strings to booleans (faster later)
    bool delete_disp = false;
    bool delete_vel = false;
    bool delete_accel = false;
    bool delete_energy = false;
    bool delete_plastic_strain = false;
    bool delete_strain = false;
    bool delete_stress = false;
    bool delete_stress_mises = false;
    bool delete_history_shell = false;
    bool delete_history_solid = false;
    for (size_t iVar = 0; iVar < _variables.size(); ++iVar) {
      if (_variables[iVar].find("disp") != std::string::npos) {
        delete_disp = true;
      } else if (_variables[iVar].find("vel") != std::string::npos) {
        delete_vel = true;
      } else if (_variables[iVar].find("accel") != std::string::npos) {
        delete_accel = true;
      } else if (_variables[iVar].find("energy") != std::string::npos) {
        delete_energy = true;
      } else if (_variables[iVar].find("plastic_strain") != std::string::npos) {
        delete_plastic_strain = true;
      } else if (_variables[iVar].find("strain") != std::string::npos) {
        delete_strain = true;
      } else if (_variables[iVar].find("stress_mises") != std::string::npos) {
        delete_stress_mises = true;
      } else if (_variables[iVar].find("stress") != std::string::npos) {
        delete_stress = true;
      } else if (_variables[iVar].find("history") != std::string::npos) {
        // shell or solid specified?
        // delete both if unspecified
        if (_variables[iVar].find("shell") != std::string::npos) {
          delete_history_shell = true;
        } else if (_variables[iVar].find("solid") != std::string::npos) {
          delete_history_solid = true;
        } else {
          delete_history_shell = true;
          delete_history_solid = true;
        }

      } else {
        throw(
          std::invalid_argument("Unknown variable type:" + _variables[iVar]));
      }

    } // end:for

    // NODES: data deletion
    if (delete_disp || delete_vel || delete_accel) {
      DB_Nodes* db_nodes = this->get_db_nodes();
      std::shared_ptr<Node> _node = nullptr;
      for (size_t iNode = 0; iNode < db_nodes->get_nNodes(); ++iNode) {
        _node = db_nodes->get_nodeByIndex(iNode);
        if (_node) {
          if (delete_disp)
            _node->clear_disp();
          if (delete_vel)
            _node->clear_vel();
          if (delete_accel)
            _node->clear_accel();
        }

      } // end:for

      // reset flags
      if (delete_disp)
        this->disp_is_read = false;
      if (delete_vel)
        this->vel_is_read = false;
      if (delete_accel)
        this->acc_is_read = false;
    }

    // ELEMENT: data deletion
    if (delete_energy || delete_plastic_strain || delete_strain ||
        delete_stress || delete_stress_mises || delete_history_shell ||
        delete_history_solid) {
      DB_Elements* db_elems = this->get_db_elements();
      std::shared_ptr<Element> _elem = nullptr;

      // shells
      for (size_t iElement = 0;
           iElement < db_elems->get_nElements(Element::SHELL);
           iElement++) {
        _elem = db_elems->get_elementByIndex(Element::SHELL, iElement);
        if (_elem) {
          if (delete_energy)
            _elem->clear_energy();
          if (delete_plastic_strain)
            _elem->clear_plastic_strain();
          if (delete_strain)
            _elem->clear_strain();
          if (delete_stress)
            _elem->clear_stress();
          if (delete_stress_mises)
            _elem->clear_stress_mises();
          if (delete_history_shell)
            _elem->clear_history_vars();
        }
      }
      // solids
      for (size_t iElement = 0;
           iElement < db_elems->get_nElements(Element::SOLID);
           iElement++) {
        _elem = db_elems->get_elementByIndex(Element::SOLID, iElement);
        if (_elem) {
          if (delete_energy)
            _elem->clear_energy();
          if (delete_plastic_strain)
            _elem->clear_plastic_strain();
          if (delete_strain)
            _elem->clear_strain();
          if (delete_stress)
            _elem->clear_stress();
          if (delete_stress_mises)
            _elem->clear_stress_mises();
          if (delete_history_solid)
            _elem->clear_history_vars();
        }
      }

      // reset flags
      if (delete_energy)
        this->energy_is_read = false;
      if (delete_plastic_strain)
        this->plastic_strain_is_read = false;
      if (delete_strain)
        this->strain_is_read = false;
      if (delete_stress)
        this->stress_is_read = false;
      if (delete_stress_mises)
        this->stress_mises_is_read = false;
      if (delete_history_shell)
        this->history_shell_is_read.clear();
      if (delete_history_solid)
        this->history_solid_is_read.clear();

    } // end:if Elements
  }   // end:else for deletion
} // end:function clear

/**
 *
 */
/*
void
D3plot::save_hdf5(const std::string& _filepath,
                 bool _overwrite_run,
                 const std::string& _run_name) const
{

 // empty string means use run title
 std::string run_folder = _run_name.empty() ? this->get_title() : _run_name;

 // hard coded settings
 H5::StrType strdatatype(H5::PredType::C_S1, 256);
 auto attr_str_dataspace = H5::DataSpace(H5S_SCALAR);

 const std::string qd_version_str(QD_VERSION);

 const std::string info_folder("/info");
 const std::string state_folder("/state_data");
 const std::string elements_folder("/elements");
 const std::string nodes_folder("/nodes");

 try {

   H5::Exception::dontPrint();

   // Open the file
   auto file = open_hdf5(_filepath, _overwrite_run);

   // info
   auto group = file.createGroup(info_folder);
   group.createAttribute("QD_VERSION", strdatatype, attr_str_dataspace)
     .write(strdatatype, qd_version_str.c_str());

 } catch (H5::FileIException error) {
   error.printError();
   throw std::runtime_error("H5::FileIException");
 }
 // catch failure caused by the DataSet operations
 catch (H5::DataSetIException error) {
   error.printError();
   throw std::runtime_error("H5::DataSetIException");
 }
 // catch failure caused by the DataSpace operations
 catch (H5::DataSpaceIException error) {
   error.printError();
   throw std::runtime_error("H5::DataSpaceIException");
 }
 // catch failure caused by the DataSpace operations
 catch (H5::DataTypeIException error) {
   error.printError();
   throw std::runtime_error("H5::DataTypeIException");
 }

} // D3plot::save_hdf5
*/

} // namespace qd
