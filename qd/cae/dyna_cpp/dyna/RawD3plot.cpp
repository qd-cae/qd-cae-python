

#include <cmath>
#include <string>

#include <dyna_cpp/dyna/D3plotBuffer.hpp>
#include <dyna_cpp/dyna/RawD3plot.hpp>
#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/MathUtility.hpp>
//#include <dyna_cpp/utility/PythonUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

#ifdef QD_USE_FEMZIP
#include "FemzipBuffer.hpp"
#endif

#ifdef QD_USE_HDF5
#include <H5Cpp.h>
#include <dyna_cpp/utility/HDF5_Utility.hpp>
#endif

namespace qd {

RawD3plot::RawD3plot(std::string _filename, bool _useFemzip)
  : dyna_ndim(-1)
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
  , wordPosition(0)
  , wordsToRead(0)
  , wordPositionStates(0)
  , useFemzip(_useFemzip)
  , femzip_state_offset(0)
  , buffer([](std::string _filename,
              bool _useFemzip) -> std::unique_ptr<AbstractBuffer> {

// WTF is this ?!?!?!
// This is a lambda for initialization of the buffer variable
// Since the buffer is a std::unique_ptr I need to do it in the
// initializer list. And since it is a little bit more complicated,
// I need to use a lambda function
#ifdef QD_USE_FEMZIP
    if (_useFemzip) {
      return std::move((std::make_unique<FemzipBuffer>(_filename)));
    } else {
      const int32_t bytesPerWord = 4;
      return std::move(std::make_unique<D3plotBuffer>(_filename, bytesPerWord));
    }
#else
    if (_useFemzip) {
      throw(std::invalid_argument(
        "d3plot.cpp was compiled without femzip support."));
    }
    const int32_t bytesPerWord = 4;
    return std::move(std::make_unique<D3plotBuffer>(_filename, bytesPerWord));
#endif

  }(_filename, _useFemzip))
{
  // --> Constructor starts here ...

  this->buffer->read_geometryBuffer(); // deallocated in read_geometry

  // Header + Geometry
  this->read_header();
  this->read_matsection();
  this->read_airbag_section();
  this->read_geometry();

  // States
  this->read_states();
}

RawD3plot::~RawD3plot()
{
#ifdef QD_DEBUG
  std::cout << "RawD3plot::~RawD3plot() called." << std::endl;
#endif
}

void
RawD3plot::read_header()
{
#ifdef QD_DEBUG
  std::cout << "> HEADER " << std::endl;
#endif

  int32_t filetype = this->buffer->read_int(11);
  if (filetype > 1000) {
    filetype -= 1000;
    own_external_numbers_I8 = true;
  }
  if ((filetype != 1) && (filetype != 5)) {
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
  // thick shells not implemented
  // if (this->dyna_nelth > 0)
  //  throw(std::runtime_error("Can not handle thick shell elements."));
  // no temps
  if (this->dyna_it != 0)
    throw(std::runtime_error("dyna_it != 0: Can not handle temperatures."));
  //
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

void
RawD3plot::info() const
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
            << "extra: " << this->dyna_extra << std::endl;
#endif
}

void
RawD3plot::read_matsection()
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
  this->dyna_irbtyp.resize({ static_cast<size_t>(tmp_nummat) });
  this->buffer->read_array<int32_t>(
    start, end - start, this->dyna_irbtyp.get_data());

  this->wordPosition += 2 + tmp_nummat;
}

void
RawD3plot::read_airbag_section()
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

void
RawD3plot::read_geometry()
{
#ifdef QD_DEBUG
  std::cout << "> GEOMETRY" << std::endl;
#endif

  /* === NODES === */
  this->node_data["coords"] = this->read_geometry_nodes();

  /* === ELEMENTS === */
  // Order MATTERS, do not swap routines.

  // 8-Node Solids
  this->elem_solids_nodes = this->read_geometry_elem8();

  // 8-Node Thick Shells
  this->elem_tshells_nodes = read_geometry_elem4th();

  // 2-Node Beams
  this->elem_beam_nodes = this->read_geometry_elem2();

  // 4-Node Elements
  this->elem_shells_nodes = this->read_geometry_elem4();

  /* === NUMBERING === */
  this->read_geometry_numbering();

  this->read_part_ids();

  /* === AIRBAGS === */
  this->read_geometry_airbag();

  // check for correct end of section
  if (!isFileEnding(wordPosition)) {
#ifdef QD_DEBUG
    std::cout << "At word position: " << wordPosition << std::endl;
#endif
    throw(
      std::runtime_error("Anticipated file ending wrong in geometry section."));
  }
  wordPosition++;

  /* === PARTS === */
  this->buffer->free_geometryBuffer();
  this->buffer->read_partBuffer();
  if (this->useFemzip)
    wordPosition = 1; // don't ask me why not 0 ...

  this->read_part_names(); // directly creates parts

  if (!isFileEnding(wordPosition)) {
#ifdef QD_DEBUG
    std::cout << "At word position: " << wordPosition << std::endl;
#endif
    throw(std::runtime_error("Anticipated file ending wrong in part section."));
  }

  this->buffer->free_partBuffer();
}

Tensor<float>
RawD3plot::read_geometry_nodes()
{
#ifdef QD_DEBUG
  std::cout << "Reading nodes at word " << wordPosition << " ... ";
#endif

  // memory to read
  wordsToRead = dyna_numnp * dyna_ndim;

  // init tensor
  Tensor<float> tensor(
    { static_cast<size_t>(dyna_numnp), static_cast<size_t>(dyna_ndim) });

  // copy stuff into tensor
  buffer->read_float_array(wordPosition, wordsToRead, tensor.get_data());

  // Update word position
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  return std::move(tensor);
}

Tensor<int32_t>
RawD3plot::read_geometry_elem8()
{
  // Check
  if (dyna_nel8 == 0)
    return Tensor<int32_t>();

#ifdef QD_DEBUG
  std::cout << "Reading solids at word " << wordPosition << " ... ";
#endif

  // currently each element has 8 nodes-ids and 1 mat-id
  const int32_t nVarsElem8 = 9;

  // allocate ids
  Tensor<int32_t> elem8_nodes(
    { static_cast<size_t>(dyna_nel8), static_cast<size_t>(nVarsElem8) });

  // do the copy stuff
  wordsToRead = nVarsElem8 * dyna_nel8;
  buffer->read_array<int32_t>(
    wordPosition, wordsToRead, elem8_nodes.get_data());

  // Update word position
  wordPosition += wordsToRead;
  if (own_nel10)
    wordPosition += 2 * dyna_nel8;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  return std::move(elem8_nodes);
}

Tensor<int32_t>
RawD3plot::read_geometry_elem4()
{
  // Check
  if (dyna_nel4 == 0)
    return Tensor<int32_t>();

#ifdef QD_DEBUG
  std::cout << "Reading shells at word " << wordPosition << " ... ";
#endif

  const int32_t nVarsElem4 = 5;

  // allocate
  Tensor<int32_t> elem4_nodes(
    { static_cast<size_t>(dyna_nel4), static_cast<size_t>(nVarsElem4) });

  // copy
  wordsToRead = nVarsElem4 * dyna_nel4;
  buffer->read_array<int32_t>(
    wordPosition, wordsToRead, elem4_nodes.get_data());

  // Update word position
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  return std::move(elem4_nodes);
}

Tensor<int32_t>
RawD3plot::read_geometry_elem2()
{
  // Check
  if (dyna_nel2 == 0)
    return Tensor<int32_t>();

#ifdef QD_DEBUG
  std::cout << "Reading beams at word " << wordPosition << " ... ";
#endif

  const int32_t nVarsElem2 = 6;

  // allocate
  Tensor<int32_t> elem2_nodes(
    { static_cast<size_t>(dyna_nel2), static_cast<size_t>(nVarsElem2) });

  // copy
  wordsToRead = nVarsElem2 * dyna_nel2;
  buffer->read_array<int32_t>(
    wordPosition, wordsToRead, elem2_nodes.get_data());

  // Update word position
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  return std::move(elem2_nodes);
}

Tensor<int32_t>
RawD3plot::read_geometry_elem4th()
{
  // Check
  if (dyna_nelth == 0)
    return Tensor<int32_t>();

#ifdef QD_DEBUG
  std::cout << "Reading thick shells at word " << wordPosition << " ... ";
#endif

  // 8 nodes and material id
  const int32_t nVarsElem4th = 9;

  // allocate
  Tensor<int32_t> elem4th_nodes(
    { static_cast<size_t>(dyna_nelth), static_cast<size_t>(nVarsElem4th) });

  // copy
  wordsToRead = nVarsElem4th * dyna_nelth;
  buffer->read_array<int32_t>(
    wordPosition, wordsToRead, elem4th_nodes.get_data());

  // Update word position
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  return std::move(elem4th_nodes);
}

void
RawD3plot::read_geometry_numbering()
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
    return;

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

  // Node IDs
  if (nsort < 0) {
    wordPosition += 16;
  } else {
    wordPosition += 10;
  }
  // wordPosition += 16; // header length is 16
  wordsToRead = dyna_numnp;
  this->node_ids.resize({ static_cast<size_t>(dyna_numnp) });
  this->buffer->read_array<int32_t>(
    wordPosition, wordsToRead, this->node_ids.get_data());

  // Solid IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nel8;
  this->elem_solids_ids.resize({ static_cast<size_t>(dyna_nel8) });
  this->buffer->read_array<int32_t>(
    wordPosition, wordsToRead, this->elem_solids_ids.get_data());

  // Beam IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nel2;
  this->elem_beams_ids.resize({ static_cast<size_t>(dyna_nel2) });
  this->buffer->read_array<int32_t>(
    wordPosition, wordsToRead, this->elem_beams_ids.get_data());

  // Shell IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nel4;
  this->elem_shells_ids.resize({ static_cast<size_t>(dyna_nel4) });
  this->buffer->read_array<int32_t>(
    wordPosition, wordsToRead, this->elem_shells_ids.get_data());

  // Thick Shell IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nelth;
  this->elem_tshells_ids.resize({ static_cast<size_t>(dyna_nelth) });
  this->buffer->read_array<int32_t>(
    wordPosition, wordsToRead, this->elem_tshells_ids.get_data());
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif
}

void
RawD3plot::read_part_ids()
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
  std::cout << "Reading part ids at word " << wordPosition << " ... ";
#endif

  // allocate
  part_ids.resize({ static_cast<size_t>(dyna_nmmat) });

  // compute memory offset
  wordsToRead = 3 * dyna_nmmat;

  // copy part numbers
  this->buffer->read_array(wordPosition, dyna_nmmat, part_ids.get_data());

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
}

void
RawD3plot::read_geometry_airbag()
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

void
RawD3plot::read_part_names()
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

  for (int32_t ii = 0; ii < this->dyna_numprop; ii++) {

    // start of the section of current part in the file
    int32_t start = (wordPosition + 1) + ii * 19 + 1;

    // this id is wrong ... and don't ask me why
    // int32_t partID = this->buffer->read_int(start);
    std::string part_name = this->buffer->read_str(start + 1, 18);
    this->part_names.push_back(part_name);
  }

  // update position
  wordPosition += 1 + (this->dyna_numprop + 1) * 19 + 1;
}

void
RawD3plot::read_states()
{

  // Calculate loop properties
  size_t iState = 0;
  int32_t nVarsNodes = dyna_ndim * (dyna_iu + dyna_iv + dyna_ia) * dyna_numnp;
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
    if ((!this->useFemzip) && firstFileDone) {
      wordPosition = 0;
    }
    // femzip case
    if (this->useFemzip) {
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
                  << std::endl;
#endif
      }

      // NODE - DISP
      if (dyna_iu)
        read_states_displacement();

      // NODE - VEL
      if (dyna_iv)
        read_states_velocity();

      // NODE - ACCEL
      if (dyna_ia)
        read_states_acceleration();

      // solids
      // read_states_elem8(iState);
      // thick shells
      // read_states_elem4th(iState);
      // shells
      // read_states_elem4(iState);

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
}

void
RawD3plot::read_states_displacement()
{

  if (dyna_iu != 1)
    return;

  const std::string var_name = "disp";

  // compute offsets
  int32_t start = wordPosition + dyna_nglbv + 1;
  wordsToRead = dyna_numnp * dyna_ndim;

  // do the magic thing
  auto& tensor = node_data[var_name];
  auto shape = tensor.get_shape();
  if (shape.size() == 0) {
    shape = { static_cast<size_t>(dyna_numnp), 0, 3 };
  }
  shape[1]++; // increase timestep count
  auto offset = tensor.size();
  tensor.resize(shape);

  this->buffer->read_array(start, wordsToRead, tensor.get_data(), offset);
}

void
RawD3plot::read_states_velocity()
{
  const std::string var_name = "vel";

  if (dyna_iv != 1)
    return;

  int32_t start =
    1 + dyna_nglbv + (dyna_iu)*dyna_numnp * dyna_ndim + wordPosition;
  wordsToRead = dyna_numnp * dyna_ndim;

  auto& tensor = node_data[var_name];
  auto shape = tensor.get_shape();
  if (shape.size() == 0) {
    shape = { static_cast<size_t>(dyna_numnp), 0, 3 };
  }
  shape[1]++; // increase timestep count
  auto offset = tensor.size();
  tensor.resize(shape);

  this->buffer->read_array(start, wordsToRead, tensor.get_data(), offset);
}

void
RawD3plot::read_states_acceleration()
{
  const std::string var_name = "accel";

  if (dyna_ia != 1)
    return;

  int32_t start = 1 + dyna_nglbv +
                  (dyna_iu + dyna_iv) * dyna_numnp * dyna_ndim + wordPosition;
  wordsToRead = dyna_numnp * dyna_ndim;
  int32_t iNode = 0;

  auto& tensor = node_data[var_name];
  auto shape = tensor.get_shape();
  if (shape.size() == 0) {
    shape = { static_cast<size_t>(dyna_numnp), 0, 3 };
  }
  shape[1]++; // increase timestep count
  auto offset = tensor.size();
  tensor.resize(shape);

  this->buffer->read_array(start, wordsToRead, tensor.get_data(), offset);
}

bool
RawD3plot::isFileEnding(int32_t iWord)
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

Tensor<float>
RawD3plot::get_node_data(const std::string& _name)
{
  return this->node_data[_name];
}

/*
std::vector<std::string>
RawD3plot::get_variables(const std::string& _name)
{

}
*/

} // namespace qd