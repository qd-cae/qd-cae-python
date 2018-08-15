

#include <cmath>
#include <string>

#include <dyna_cpp/dyna/d3plot/D3plotBuffer.hpp>
#include <dyna_cpp/dyna/d3plot/RawD3plot.hpp>
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

RawD3plot::RawD3plot()
  : dyna_filetype(-1)
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
  , dyna_irbtyp(std::make_shared<Tensor<int32_t>>())
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
  , own_nDeletionVars(0)
  , wordPosition(0)
  , wordsToRead(0)
  , wordPositionStates(0)
  , femzip_state_offset(0)
  , buffer(nullptr)
{}

RawD3plot::RawD3plot(std::string _filename, bool use_femzip)
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
  , dyna_irbtyp(std::make_shared<Tensor<int32_t>>())
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
  , own_nDeletionVars(0)
  , wordPosition(0)
  , wordsToRead(0)
  , wordPositionStates(0)
  , _is_femzipped(false)
  , femzip_state_offset(0)
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
  dyna_irbtyp->resize({ static_cast<size_t>(tmp_nummat) });
  this->buffer->read_array<int32_t>(start, tmp_nummat, dyna_irbtyp->get_data());
  int_data.insert(std::make_pair("material_type_numbers", dyna_irbtyp));

  this->wordPosition += 2 + tmp_nummat;
}

void
RawD3plot::read_airbag_section()
{

#ifdef QD_DEBUG
  std::cout << "> AIRBAG (PARTICLES)" << std::endl;
#endif

  // skip airbag particle section
  if ((dyna_npefg > 0) && (dyna_npefg < 10000000)) {

    dyna_airbag_npartgas = dyna_npefg % 1000; // nAirbags?!
    dyna_airbag_subver = dyna_npefg / 1000;   // n

    dyna_airbag_ngeom = this->buffer->read_int(wordPosition++);
    dyna_airbag_state_nvars = this->buffer->read_int(wordPosition++);
    dyna_airbag_nparticles = this->buffer->read_int(wordPosition++);
    dyna_airbag_state_geom = this->buffer->read_int(wordPosition++);
    dyna_airbag_nchamber = 0;
    if (dyna_airbag_subver == 4) {
      dyna_airbag_nchamber = this->buffer->read_int(wordPosition++);
    }

    // read nlist (type of state vars)
    int32_t dyna_airbag_nlist_size =
      dyna_airbag_ngeom + dyna_airbag_state_nvars + dyna_airbag_state_geom;

    dyna_airbag_nlist.resize(dyna_airbag_nlist_size);
    this->buffer->read_array(
      wordPosition, dyna_airbag_nlist_size, dyna_airbag_nlist);

    auto tensor_nlist = std::make_shared<Tensor<int32_t>>();
    int_data.insert(std::make_pair("airbag_variable_type_flag", tensor_nlist));
    tensor_nlist->resize({ dyna_airbag_nlist.size() });
    std::copy(dyna_airbag_nlist.begin(),
              dyna_airbag_nlist.end(),
              tensor_nlist->get_data().begin());

    wordPosition +=
      dyna_airbag_nlist_size; // type of each variable (1=int, 2=float)

    // nlist variable names (1 Word is a char ... whoever had that idea ...)
    auto& airbag_all_variable_names = string_data["airbag_all_variable_names"];
    airbag_all_variable_names.resize(dyna_airbag_nlist_size);

    std::vector<int32_t> char_buffer(8 * dyna_airbag_nlist_size);
    this->buffer->read_array(
      wordPosition, 8 * dyna_airbag_nlist_size, char_buffer);

    std::string _tmp_string;
    _tmp_string.resize(8);

    for (size_t iString = 0;
         iString < static_cast<size_t>(dyna_airbag_nlist_size);
         ++iString) {
      for (size_t iChar = 0; iChar < 8; ++iChar)
        _tmp_string[iChar] = (char)char_buffer[iString * 8 + iChar];
      airbag_all_variable_names[iString] = _tmp_string;
    }

    wordPosition += 8 * dyna_airbag_nlist_size; // 8 char variable names

    // resort variable names into categories
    auto& airbag_geom_names = string_data["airbag_geom_names"];
    auto& airbag_geom_int_names = string_data["airbag_geom_state_int_names"];
    auto& airbag_geom_float_names =
      string_data["airbag_geom_state_float_names"];
    auto& airbag_particle_int_var_names =
      string_data["airbag_particle_int_names"];
    auto& airbag_particle_float_var_names =
      string_data["airbag_particle_float_names"];

    // geometry var names
    for (int32_t ii = 0; ii < dyna_airbag_ngeom; ++ii)
      airbag_geom_names.push_back(airbag_all_variable_names[ii]);

    // particle var names
    for (size_t ii = static_cast<size_t>(dyna_airbag_ngeom);
         ii < dyna_airbag_nlist.size() - dyna_airbag_state_geom;
         ++ii) {
      if (dyna_airbag_nlist[ii] == 1)
        airbag_particle_int_var_names.push_back(airbag_all_variable_names[ii]);
      else if (dyna_airbag_nlist[ii] == 2)
        airbag_particle_float_var_names.push_back(
          airbag_all_variable_names[ii]);
      else
        throw(
          std::runtime_error("dyna_airbag_nlist != 1 or 2 makes no sense."));
    }

    // state geometry var names
    for (size_t ii = static_cast<size_t>(dyna_airbag_ngeom) +
                     static_cast<size_t>(dyna_airbag_state_nvars);
         ii < dyna_airbag_nlist.size();
         ++ii) {
      if (dyna_airbag_nlist[ii] == 1)
        airbag_geom_int_names.push_back(airbag_all_variable_names[ii]);
      else if (dyna_airbag_nlist[ii] == 2)
        airbag_geom_float_names.push_back(airbag_all_variable_names[ii]);
      else
        throw(
          std::runtime_error("dyna_airbag_nlist != 1 or 2 makes no sense."));
    }

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
              << "dyna_airbag_nlist_size: " << dyna_airbag_nlist_size
              << std::endl;
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
  this->read_geometry_nodes();

  /* === ELEMENTS === */
  // Order MATTERS, do not swap routines.

  // 8-Node Solids
  this->read_geometry_elem8();

  // 8-Node Thick Shells
  this->read_geometry_elem4th();

  // 2-Node Beams
  this->read_geometry_elem2();

  // 4-Node Elements
  this->read_geometry_elem4();

  /* === NUMBERING === */
  this->read_geometry_numbering();

  this->read_part_ids();

  /* === AIRBAGS === */
  this->read_geometry_airbag();

  // check for correct end of section
  if (isFileEnding(wordPosition)) {
#ifdef QD_DEBUG
    std::cout << "At word position: " << wordPosition << std::endl;
#endif
    // throw(
    //   std::runtime_error("Anticipated file ending wrong in geometry
    //   section."));

    wordPosition++;

    /* === PARTS === */
    this->buffer->free_geometryBuffer();
    this->buffer->read_partBuffer();
    if (this->_is_femzipped)
      wordPosition = 1; // don't ask me why not 0 ...

    this->read_part_names(); // directly creates parts

    if (!isFileEnding(wordPosition)) {
#ifdef QD_DEBUG
      std::cout << "At word position: " << wordPosition << std::endl;
#endif
      throw(
        std::runtime_error("Anticipated file ending wrong in part section."));
    }

    this->buffer->free_partBuffer();
  }
}

void
RawD3plot::read_geometry_nodes()
{
#ifdef QD_DEBUG
  std::cout << "Reading nodes at word " << wordPosition << " ... ";
#endif

  if (dyna_numnp < 1)
    return;

  // memory to read
  wordsToRead = dyna_numnp * dyna_ndim;

  // init tensor
  auto tensor = std::make_shared<Tensor<float>>();
  float_data.insert(std::make_pair("node_coordinates", tensor));
  tensor->resize(
    { static_cast<size_t>(dyna_numnp), static_cast<size_t>(dyna_ndim) });

  // copy stuff into tensor
  buffer->read_float_array(wordPosition, wordsToRead, tensor->get_data());

  // Update word position
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif
}

void
RawD3plot::read_geometry_elem8()
{
  // Check
  if (dyna_nel8 == 0)
    return;

#ifdef QD_DEBUG
  std::cout << "Reading solids at word " << wordPosition << " ... ";
#endif

  // currently each element has 8 nodes-ids and 1 mat-id
  const int32_t nVarsElem8 = 9;

  // allocate ids
  auto elem8_nodes = std::make_shared<Tensor<int32_t>>();
  int_data.insert(std::make_pair("elem_solid_data", elem8_nodes));
  elem8_nodes->resize(
    { static_cast<size_t>(dyna_nel8), static_cast<size_t>(nVarsElem8) });

  // do the copy stuff
  wordsToRead = nVarsElem8 * dyna_nel8;
  buffer->read_array<int32_t>(
    wordPosition, wordsToRead, elem8_nodes->get_data());

  // Update word position
  wordPosition += wordsToRead;
  if (own_nel10)
    wordPosition += 2 * dyna_nel8;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif
}

void
RawD3plot::read_geometry_elem4()
{
  // Check
  if (dyna_nel4 == 0)
    return;

#ifdef QD_DEBUG
  std::cout << "Reading shells at word " << wordPosition << " ... ";
#endif

  const int32_t nVarsElem4 = 5;

  // allocate
  auto elem4_nodes = std::make_shared<Tensor<int32_t>>();
  int_data.insert(std::make_pair("elem_shell_data", elem4_nodes));
  elem4_nodes->resize(
    { static_cast<size_t>(dyna_nel4), static_cast<size_t>(nVarsElem4) });

  // copy
  wordsToRead = nVarsElem4 * dyna_nel4;
  buffer->read_array<int32_t>(
    wordPosition, wordsToRead, elem4_nodes->get_data());

  // Update word position
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif
}

void
RawD3plot::read_geometry_elem2()
{
  // Check
  if (dyna_nel2 == 0)
    return;

#ifdef QD_DEBUG
  std::cout << "Reading beams at word " << wordPosition << " ... ";
#endif

  const int32_t nVarsElem2 = 6;

  // allocate
  auto elem2_nodes = std::make_shared<Tensor<int32_t>>();
  int_data.insert(std::make_pair("elem_beam_data", elem2_nodes));
  elem2_nodes->resize(
    { static_cast<size_t>(dyna_nel2), static_cast<size_t>(nVarsElem2) });

  // copy
  wordsToRead = nVarsElem2 * dyna_nel2;
  buffer->read_array<int32_t>(
    wordPosition, wordsToRead, elem2_nodes->get_data());

  // Update word position
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif
}

void
RawD3plot::read_geometry_elem4th()
{
  // Check
  if (dyna_nelth == 0)
    return;

#ifdef QD_DEBUG
  std::cout << "Reading thick shells at word " << wordPosition << " ... ";
#endif

  // 8 nodes and material id
  const int32_t nVarsElem4th = 9;

  // allocate
  auto elem4th_nodes = std::make_shared<Tensor<int32_t>>();
  int_data.insert(std::make_pair("elem_tshell_data", elem4th_nodes));
  elem4th_nodes->resize(
    { static_cast<size_t>(dyna_nelth), static_cast<size_t>(nVarsElem4th) });

  // copy
  wordsToRead = nVarsElem4th * dyna_nelth;
  buffer->read_array<int32_t>(
    wordPosition, wordsToRead, elem4th_nodes->get_data());

  // Update word position
  wordPosition += wordsToRead;

#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif
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
  if (dyna_numnp > 0) {
    auto node_ids = std::make_shared<Tensor<int32_t>>();
    int_data.insert(std::make_pair("node_ids", node_ids));
    node_ids->resize({ static_cast<size_t>(dyna_numnp) });
    this->buffer->read_array<int32_t>(
      wordPosition, wordsToRead, node_ids->get_data());
  }

  // Solid IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nel8;
  if (dyna_nel8 > 0) {
    auto elem_solid_ids = std::make_shared<Tensor<int32_t>>();
    int_data.insert(std::make_pair("elem_solid_ids", elem_solid_ids));
    elem_solid_ids->resize({ static_cast<size_t>(dyna_nel8) });
    this->buffer->read_array<int32_t>(
      wordPosition, wordsToRead, elem_solid_ids->get_data());
  }

  // Beam IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nel2;
  if (dyna_nel2 > 0) {
    auto elem_beam_ids = std::make_shared<Tensor<int32_t>>();
    int_data.insert(std::make_pair("elem_beam_ids", elem_beam_ids));
    elem_beam_ids->resize({ static_cast<size_t>(dyna_nel2) });
    this->buffer->read_array<int32_t>(
      wordPosition, wordsToRead, elem_beam_ids->get_data());
  }

  // Shell IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nel4;
  if (dyna_nel4 > 0) {
    auto elem_shell_ids = std::make_shared<Tensor<int32_t>>();
    int_data.insert(std::make_pair("elem_shell_ids", elem_shell_ids));
    elem_shell_ids->resize({ static_cast<size_t>(dyna_nel4) });
    this->buffer->read_array<int32_t>(
      wordPosition, wordsToRead, elem_shell_ids->get_data());
  }

  // Thick Shell IDs
  wordPosition += wordsToRead;
  wordsToRead = dyna_nelth;
  if (dyna_nelth > 0) {
    auto elem_tshell_ids = std::make_shared<Tensor<int32_t>>();
    int_data.insert(std::make_pair("elem_tshell_ids", elem_tshell_ids));
    elem_tshell_ids->resize({ static_cast<size_t>(dyna_nelth) });
    this->buffer->read_array<int32_t>(
      wordPosition, wordsToRead, elem_tshell_ids->get_data());
  }

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
  auto part_ids = std::make_shared<Tensor<int32_t>>();
  int_data.insert(std::make_pair("part_ids", part_ids));
  part_ids->resize({ static_cast<size_t>(dyna_nmmat) });

  // compute memory offset
  wordsToRead = 3 * dyna_nmmat;

  // copy part numbers
  this->buffer->read_array(wordPosition, dyna_nmmat, part_ids->get_data());

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

/** Read the geometry airbag data
 *
 */
void
RawD3plot::read_geometry_airbag()
{

  if (this->dyna_npefg > 0) {

    // ?!?!?!?!
    if (this->dyna_npefg / 10000000 == 1) {
      int32_t dyna_airbag_des = this->dyna_npefg / 10000000;
      wordPosition += dyna_airbag_des;
    }

    auto tensor = std::make_shared<Tensor<int32_t>>();
    int_data.insert(std::make_pair("airbag_geometry", tensor));
    tensor->resize({ static_cast<size_t>(dyna_airbag_npartgas),
                     static_cast<size_t>(dyna_airbag_ngeom) });

    this->buffer->read_array(wordPosition,
                             dyna_airbag_npartgas * dyna_airbag_ngeom,
                             tensor->get_data());

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

/** Read the part names from the d3plot
 *
 */
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

  auto& part_names = this->string_data["part_names"];
  for (int32_t ii = 0; ii < this->dyna_numprop; ii++) {

    // start of the section of current part in the file
    int32_t start = (wordPosition + 1) + ii * 19 + 1;

    // this id is wrong ... and don't ask me why
    // int32_t partID = this->buffer->read_int(start);
    std::string part_name = this->buffer->read_str(start + 1, 18);
    part_names.push_back(part_name);
  }

  // update position
  wordPosition += 1 + (this->dyna_numprop + 1) * 19 + 1;
}

/** Read all states in the d3plot
 *
 */
void
RawD3plot::read_states()
{

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
  if (dyna_mdlopt == 0) {
    // ok
  } else if (dyna_mdlopt == 1) {
    own_nDeletionVars = dyna_numnp;
  } else if (dyna_mdlopt == 2) {
    own_nDeletionVars = dyna_nel2 + dyna_nel4 + dyna_nel8 + dyna_nelth;
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
                  << std::endl;
#endif
      }

      // NODE - DISP
      if (dyna_iu)
        read_states_displacement();

      // NODE - MASS SCALING
      if (own_has_mass_scaling_info)
        read_states_nodes_mass_scaling();

      // NODE - VEL
      if (dyna_iv)
        read_states_velocity();

      // NODE - ACCEL
      if (dyna_ia)
        read_states_acceleration();

      // solids
      read_states_elem8();

      // thick shells
      read_states_elem4th();

      //  beams
      read_states_elem2();

      // shells
      read_states_elem4();

      // element deletion info
      read_states_elem_deletion();

      // airbag
      read_states_airbag();

      // update position
      // +1 is just for time word
      wordPosition += nAirbagVars + nVarsNodes + nVarsElems +
                      own_nDeletionVars + dyna_nglbv + 1;

      iState++;
    }

    firstFileDone = true;
  }

  this->buffer->end_nextState();

  // save some state variable data
  auto timesteps_tensor = std::make_shared<Tensor<float>>();
  float_data.insert(std::make_pair("timesteps", timesteps_tensor));
  timesteps_tensor->resize({ timesteps.size() });
  std::copy(
    timesteps.begin(), timesteps.end(), timesteps_tensor->get_data().begin());
}

/** Read the node mass scaling ifo
 *
 * How long does it take an engineer to find out someone is writing
 * mass scaling between disp and vel field ... I can tell u ...
 */
void
RawD3plot::read_states_nodes_mass_scaling()
{

  if (!own_has_mass_scaling_info)
    return;

  const std::string var_name = "node_mass_scaling";

  // compute offsets
  int32_t start = this->wordPosition + 1 // time
                  + dyna_nglbv + dyna_iu * dyna_ndim * dyna_numnp;

  wordsToRead = dyna_numnp;

#ifdef QD_DEBUG
  std::cout << "> read_states_nodes_mass_scaling at " << start << std::endl;
#endif

  // do the magic thing
  auto tensor = std::make_shared<Tensor<float>>();
  float_data.insert(std::make_pair(var_name, tensor));
  auto shape = tensor->get_shape();
  if (shape.size() == 0) {
    shape = { 0, static_cast<size_t>(dyna_numnp) };
  }
  shape[0]++; // increase timestep count
  auto offset = tensor->size();
  tensor->resize(shape);

  this->buffer->read_array(start, wordsToRead, tensor->get_data(), offset);
}

/** Read the nodal displacement in the state section
 *
 */
void
RawD3plot::read_states_displacement()
{

  if (dyna_iu != 1)
    return;

  const std::string var_name = "node_displacement";

  // compute offsets
  int32_t start = wordPosition + dyna_nglbv + 1;

  wordsToRead = dyna_numnp * dyna_ndim;

#ifdef QD_DEBUG
  std::cout << "> read_states_displacement at " << start << std::endl;
#endif

  // do the magic thing
  std::shared_ptr<Tensor<float>> tensor = nullptr;
  if (float_data.find(var_name) == float_data.end()) {
    tensor = std::make_shared<Tensor<float>>();
    float_data.insert(std::make_pair(var_name, tensor));
  } else {
    tensor = float_data[var_name];
  }
  auto shape = tensor->get_shape();
  if (shape.size() == 0) {
    shape = { 0, static_cast<size_t>(dyna_numnp), 3 };
  }
  shape[0]++; // increase timestep count
  auto offset = tensor->size();
  tensor->resize(shape);

  this->buffer->read_array(start, wordsToRead, tensor->get_data(), offset);
}

/** Read the nodal velocity in the state section
 *
 */
void
RawD3plot::read_states_velocity()
{

  const std::string var_name = "node_velocity";

  if (dyna_iv != 1)
    return;

  int32_t start =
    wordPosition + 1 + dyna_nglbv +
    (dyna_iu * dyna_ndim + own_has_mass_scaling_info) * dyna_numnp;

  wordsToRead = dyna_numnp * dyna_ndim;

#ifdef QD_DEBUG
  std::cout << "> read_states_velocity at " << start << std::endl;
#endif

  std::shared_ptr<Tensor<float>> tensor = nullptr;
  if (float_data.find(var_name) == float_data.end()) {
    tensor = std::make_shared<Tensor<float>>();
    float_data.insert(std::make_pair(var_name, tensor));
  } else {
    tensor = float_data[var_name];
  }
  auto shape = tensor->get_shape();
  if (shape.size() == 0) {
    shape = { 0, static_cast<size_t>(dyna_numnp), 3 };
  }
  shape[0]++; // increase timestep count
  auto offset = tensor->size();
  tensor->resize(shape);

  this->buffer->read_array(start, wordsToRead, tensor->get_data(), offset);
}

/** Read the nodal acceleration in the state section
 *
 */
void
RawD3plot::read_states_acceleration()
{

  const std::string var_name = "node_acceleration";

  if (dyna_ia != 1)
    return;

  int32_t start =
    wordPosition + 1 + dyna_nglbv +
    ((dyna_iu + dyna_iv) * dyna_ndim + own_has_mass_scaling_info) * dyna_numnp;
  wordsToRead = dyna_numnp * dyna_ndim;

#ifdef QD_DEBUG
  std::cout << "> read_states_acceleration at " << start << std::endl;
#endif

  std::shared_ptr<Tensor<float>> tensor = nullptr;
  if (float_data.find(var_name) == float_data.end()) {
    tensor = std::make_shared<Tensor<float>>();
    float_data.insert(std::make_pair(var_name, tensor));
  } else {
    tensor = float_data[var_name];
  }
  auto shape = tensor->get_shape();
  if (shape.size() == 0) {
    shape = { 0, static_cast<size_t>(dyna_numnp), 3 };
  }
  shape[0]++; // increase timestep count
  auto offset = tensor->size();
  tensor->resize(shape);

  this->buffer->read_array(start, wordsToRead, tensor->get_data(), offset);
}

/** Read the state variables for solid elements
 *
 */
void
RawD3plot::read_states_elem8()
{

  const std::string var_name = "elem_solid_results";

  if ((dyna_nv3d <= 0) || (dyna_nel8 <= 0))
    return;

  int32_t start =
    this->wordPosition + 1 // time
    + dyna_nglbv +
    ((dyna_iu + dyna_iv + dyna_ia) * dyna_ndim + own_has_mass_scaling_info) *
      dyna_numnp;

  wordsToRead = dyna_nv3d * dyna_nel8;

#ifdef QD_DEBUG
  std::cout << "> read_states_elem8 at " << start << std::endl;
#endif

  // allocate
  std::shared_ptr<Tensor<float>> tensor = nullptr;
  if (float_data.find(var_name) == float_data.end()) {
    tensor = std::make_shared<Tensor<float>>();
    float_data.insert(std::make_pair(var_name, tensor));
  } else {
    tensor = float_data[var_name];
  }
  auto shape = tensor->get_shape();
  if (shape.size() == 0) {
    shape = {
      0,
      static_cast<size_t>(dyna_nel8),
      static_cast<size_t>(dyna_nv3d),
    };
  }
  shape[0]++; // one more timestep
  auto offset = tensor->size();
  tensor->resize(shape);

  // read
  this->buffer->read_array(start, wordsToRead, tensor->get_data(), offset);
}

/** Read the state variables for shell elements
 *
 */
void
RawD3plot::read_states_elem4()
{

  if ((dyna_nv2d <= 0) || (dyna_nel4 - dyna_numrbe <= 0))
    return;

  // prepare looping
  int32_t start =
    this->wordPosition + 1 // time
    + dyna_nglbv +
    ((dyna_iu + dyna_iv + dyna_ia) * dyna_ndim + own_has_mass_scaling_info) *
      dyna_numnp +
    dyna_nv3d * dyna_nel8 + dyna_nelth * dyna_nv3dt + dyna_nv1d * dyna_nel2;

  wordsToRead = dyna_nv2d * (dyna_nel4 - dyna_numrbe);

#ifdef QD_DEBUG
  std::cout << "> read_states_elem4 at " << start << std::endl;
#endif

  // offsets
  int32_t iPlastStrainOffset = this->dyna_ioshl1 * 6; // stresses before?
  int32_t iHistoryOffset =
    iPlastStrainOffset + this->dyna_ioshl2; // stresses & pl. strain before
  int32_t iLayerSize = dyna_neips + iHistoryOffset;

  int32_t nLayerVars = dyna_maxint * iLayerSize;
  size_t nLayerVars_unsigned = static_cast<size_t>(nLayerVars);
  int32_t nNormalVars = dyna_nv2d - dyna_maxint * iLayerSize;
  size_t nNormalVars_unsigned = static_cast<size_t>(nNormalVars);

  // allocate
  std::shared_ptr<Tensor<float>> shell_layer_vars = nullptr;
  if (float_data.find("elem_shell_results_layers") == float_data.end()) {
    shell_layer_vars = std::make_shared<Tensor<float>>();
    float_data.insert(
      std::make_pair("elem_shell_results_layers", shell_layer_vars));
  } else {
    shell_layer_vars = float_data["elem_shell_results_layers"];
  }
  auto shape = shell_layer_vars->get_shape();
  if (shape.size() == 0) {
    shape = { 0,
              static_cast<size_t>(dyna_nel4 - dyna_numrbe),
              static_cast<size_t>(dyna_maxint),
              static_cast<size_t>(iLayerSize) };
  }
  shape[0]++; // one more timestep
  auto offset_shell_layer_vars = shell_layer_vars->size();
  shell_layer_vars->resize(shape);

  std::shared_ptr<Tensor<float>> shell_vars = nullptr;
  if (float_data.find("elem_shell_results") == float_data.end()) {
    shell_vars = std::make_shared<Tensor<float>>();
    float_data.insert(std::make_pair("elem_shell_results", shell_vars));
  } else {
    shell_vars = float_data["elem_shell_results"];
  }

  auto shape2 = shell_vars->get_shape();
  if (shape2.size() == 0) {
    shape2 = {
      0,
      static_cast<size_t>(dyna_nel4 - dyna_numrbe),
      static_cast<size_t>(nNormalVars),
    };
  }
  shape2[0]++; // one more timestep
  auto offset_shell_vars = shell_vars->size();
  shell_vars->resize(shape2);

  // Do the thing ...
  for (int32_t ii = start; ii < start + wordsToRead; ii += dyna_nv2d) {

    this->buffer->read_array(
      ii, nLayerVars, shell_layer_vars->get_data(), offset_shell_layer_vars);
    offset_shell_layer_vars += nLayerVars_unsigned;

    this->buffer->read_array(
      ii + nLayerVars, nNormalVars, shell_vars->get_data(), offset_shell_vars);
    offset_shell_vars += nNormalVars_unsigned;

  } // for elements
}

/** Read the state data of the thick shell elements
 *
 */
void
RawD3plot::read_states_elem4th()
{

  if ((dyna_nv3dt <= 0) || (dyna_nelth <= 0))
    return;

  // prepare looping
  int32_t start =
    this->wordPosition + 1 // time
    + dyna_nglbv +
    ((dyna_iu + dyna_iv + dyna_ia) * dyna_ndim + own_has_mass_scaling_info) *
      dyna_numnp +
    dyna_nv3d * dyna_nel8; // solids

  wordsToRead = dyna_nelth * dyna_nv3dt;

#ifdef QD_DEBUG
  std::cout << "> read_states_elem4th at " << start << std::endl;
#endif

  // offsets
  int32_t iPlastStrainOffset = this->dyna_ioshl1 * 6; // stresses before?
  int32_t iHistoryOffset =
    iPlastStrainOffset + this->dyna_ioshl2; // stresses & pl. strain before
  int32_t iLayerSize = dyna_neips + iHistoryOffset;

  int32_t nLayerVars = dyna_maxint * iLayerSize;
  size_t nLayerVars_unsigned = static_cast<size_t>(nLayerVars);
  int32_t nNormalVars = dyna_nv3dt - dyna_maxint * iLayerSize;
  size_t nNormalVars_unsigned = static_cast<size_t>(nNormalVars);

  // allocate
  std::shared_ptr<Tensor<float>> tshell_layer_vars = nullptr;
  if (float_data.find("elem_tshell_results_layers") == float_data.end()) {
    tshell_layer_vars = std::make_shared<Tensor<float>>();
    float_data.insert(
      std::make_pair("elem_tshell_results_layers", tshell_layer_vars));
  } else {
    tshell_layer_vars = float_data["elem_tshell_results_layers"];
  }
  auto shape = tshell_layer_vars->get_shape();
  if (shape.size() == 0) {
    shape = { 0,
              static_cast<size_t>(dyna_nelth),
              static_cast<size_t>(dyna_maxint),
              static_cast<size_t>(iLayerSize) };
  }
  shape[0]++; // one more timestep
  auto offset_tshell_layer_vars = tshell_layer_vars->size();
  tshell_layer_vars->resize(shape);

  std::shared_ptr<Tensor<float>> tshell_vars = nullptr;
  if (float_data.find("elem_tshell_results") == float_data.end()) {
    tshell_vars = std::make_shared<Tensor<float>>();
    float_data.insert(std::make_pair("elem_tshell_results", tshell_vars));
  } else {
    tshell_vars = float_data["elem_tshell_results"];
  }
  auto shape2 = tshell_vars->get_shape();
  if (shape2.size() == 0) {
    shape2 = {
      0,
      static_cast<size_t>(dyna_nelth),
      static_cast<size_t>(nNormalVars),
    };
  }
  shape2[0]++; // one more timestep
  auto offset_tshell_vars = tshell_vars->size();
  tshell_vars->resize(shape2);

  // Do the thing ...
  for (int32_t ii = start; ii < start + wordsToRead; ii += dyna_nv3dt) {

    this->buffer->read_array(
      ii, nLayerVars, tshell_layer_vars->get_data(), offset_tshell_layer_vars);
    offset_tshell_layer_vars += nLayerVars_unsigned;

    this->buffer->read_array(ii + nLayerVars,
                             nNormalVars,
                             tshell_vars->get_data(),
                             offset_tshell_vars);
    offset_tshell_vars += nNormalVars_unsigned;

  } // for elements
}

/** Read the state variables for beam elements
 *
 */
void
RawD3plot::read_states_elem2()
{

  if ((dyna_nv2d <= 0) || (dyna_nel2 <= 0))
    return;

  // prepare looping
  int32_t start =
    this->wordPosition + 1 // time
    + dyna_nglbv +
    ((dyna_iu + dyna_iv + dyna_ia) * dyna_ndim + own_has_mass_scaling_info) *
      dyna_numnp +
    dyna_nv3d * dyna_nel8 + dyna_nelth * dyna_nv3dt;

  wordsToRead = dyna_nv1d * dyna_nel2;

#ifdef QD_DEBUG
  std::cout << "> read_states_elem2 at " << start << std::endl;
#endif

  const std::string var_name = "elem_beam_results";

  // allocate
  std::shared_ptr<Tensor<float>> tensor = nullptr;
  if (float_data.find(var_name) == float_data.end()) {
    tensor = std::make_shared<Tensor<float>>();
    float_data.insert(std::make_pair(var_name, tensor));
  } else {
    tensor = float_data[var_name];
  }
  auto shape = tensor->get_shape();
  if (shape.size() == 0) {
    shape = {
      0,
      static_cast<size_t>(dyna_nel2),
      static_cast<size_t>(dyna_nv2d),
    };
  }
  shape[0]++; // one more timestep
  auto offset = tensor->size();
  tensor->resize(shape);

  // read
  this->buffer->read_array(start, wordsToRead, tensor->get_data(), offset);
}

/** Read the deletion variables for elements (or nodes)
 *
 */
void
RawD3plot::read_states_elem_deletion()
{

  int32_t start = this->wordPosition + 1 // time
                  + dyna_nglbv +
                  (dyna_iu + dyna_iv + dyna_ia) * dyna_numnp * dyna_ndim +
                  dyna_nv3d * dyna_nel8 + dyna_nelth * dyna_nv3dt +
                  dyna_nv1d * dyna_nel2 + dyna_nv2d * (dyna_nel4 - dyna_numrbe);

  // Node deletion info
  if (dyna_mdlopt == 1) {

    const std::string var_name = "node_deletion_info";

    std::shared_ptr<Tensor<float>> tensor = nullptr;
    if (float_data.find(var_name) == float_data.end()) {
      tensor = std::make_shared<Tensor<float>>();
      float_data.insert(std::make_pair(var_name, tensor));
    } else {
      tensor = float_data[var_name];
    }

    auto shape = tensor->get_shape();
    if (shape.size() == 0)
      shape = { 0, static_cast<size_t>(dyna_numnp) };

    shape[0]++; // one more timestep
    auto offset = tensor->size();
    tensor->resize(shape);

    // read
    this->buffer->read_array(start, dyna_numnp, tensor->get_data(), offset);

  }
  // Element deletion info
  else if (dyna_mdlopt == 2) {

    // field names hard coded (yay)
    const std::vector<std::string> var_names = { "elem_solid_deletion_info",
                                                 "elem_tshell_deletion_info",
                                                 "elem_shell_deletion_info",
                                                 "elem_beam_deletion_info" };
    const std::vector<int32_t> var_field_sizes = {
      dyna_nel8, dyna_nelth, dyna_nel4, dyna_nel2
    };

    // read fields
    for (size_t ii = 0; ii < var_names.size(); ++ii) {

      auto field_name = var_names[ii];
      wordsToRead = var_field_sizes[ii];
      if (wordsToRead < 1)
        continue;

      std::shared_ptr<Tensor<float>> tensor = nullptr;
      if (float_data.find(field_name) == float_data.end()) {
        tensor = std::make_shared<Tensor<float>>();
        float_data.insert(std::make_pair(field_name, tensor));
      } else {
        tensor = float_data[field_name];
      }

      auto shape = tensor->get_shape();
      if (shape.size() == 0)
        shape = { 0, static_cast<size_t>(wordsToRead) };

      shape[0]++; // one more timestep
      auto offset = tensor->size();
      tensor->resize(shape);

      // read
      this->buffer->read_array(start, wordsToRead, tensor->get_data(), offset);
      start += wordsToRead;
    }

  }
  // Error
  else {
    throw(std::runtime_error("dyna_mdlopt=" + std::to_string(dyna_mdlopt) +
                             " makes no sense and must be 1 or 2."));
  }
}

/** Read the state variables for airbags
 *
 * (currently disabled)
 */
void
RawD3plot::read_states_airbag()
{

  if ((dyna_airbag_nparticles <= 0) || (dyna_airbag_state_nvars <= 0))
    return;

  int32_t start =
    this->wordPosition + 1 // time
    + dyna_nglbv + (dyna_iu + dyna_iv + dyna_ia) * dyna_numnp * dyna_ndim +
    dyna_nv3d * dyna_nel8 + dyna_nelth * dyna_nv3dt + dyna_nv1d * dyna_nel2 +
    dyna_nv2d * (dyna_nel4 - dyna_numrbe) + own_nDeletionVars;

#ifdef QD_DEBUG
  std::cout << "> read_states_airbag at " << start << std::endl;
#endif

  // dyna_nlist and airbag_all_variable_names have the same length
  // the order of variables should be:
  //
  // geometry vars -> particle state vars -> airbag state vars
  //
  // oh and this is not documented anywhere ...

  // count ints and floats for geometry state vars
  size_t nIntegerVars_state_geom = 0;
  size_t nFloatVars_state_geom = 0;
  for (size_t ii =
         dyna_airbag_nlist.size() - static_cast<size_t>(dyna_airbag_state_geom);
       ii < dyna_airbag_nlist.size();
       ++ii) {
    if (dyna_airbag_nlist[ii] == 1)
      ++nIntegerVars_state_geom;
    else
      ++nFloatVars_state_geom;
  }

  // count ints and floats for particle state vars
  size_t nIntegerVars_particles = 0;
  size_t nFloatVars_particles = 0;
  for (size_t ii = static_cast<size_t>(dyna_airbag_ngeom);
       ii <
       dyna_airbag_nlist.size() - static_cast<size_t>(dyna_airbag_state_geom);
       ++ii) {
    if (dyna_airbag_nlist[ii] == 1)
      ++nIntegerVars_particles;
    else
      ++nFloatVars_particles;
  }

  // AIRBAG GEOMETRY STATE DATA

  // for dyna_airbag_npartgas
  // 1. number of active particles
  // 2. current bag volume
  wordsToRead = dyna_airbag_npartgas * dyna_airbag_state_geom;

  // allocate
  const std::string tensor_float_name = "airbag_geom_state_float_results";
  std::shared_ptr<Tensor<float>> tensor_float = nullptr;
  if (float_data.find(tensor_float_name) == float_data.end()) {
    tensor_float = std::make_shared<Tensor<float>>();
    float_data.insert(std::make_pair(tensor_float_name, tensor_float));
  } else {
    tensor_float = float_data[tensor_float_name];
  }
  auto shape = tensor_float->get_shape();
  if (shape.size() == 0) {
    shape = { 0,
              static_cast<size_t>(dyna_airbag_npartgas),
              nFloatVars_state_geom };
  }
  shape[0]++; // one more timestep
  auto float_offset = tensor_float->size();
  tensor_float->resize(shape);
  auto& tensor_float_vec = tensor_float->get_data();

  const std::string tensor_int_name = "airbag_geom_state_int_results";
  std::shared_ptr<Tensor<int32_t>> tensor_int = nullptr;
  if (int_data.find(tensor_int_name) == int_data.end()) {
    tensor_int = std::make_shared<Tensor<int32_t>>();
    int_data.insert(std::make_pair(tensor_int_name, tensor_int));
  } else {
    tensor_int = int_data[tensor_int_name];
  }
  shape = tensor_int->get_shape();
  if (shape.size() == 0) {
    shape = { 0,
              static_cast<size_t>(dyna_airbag_npartgas),
              nIntegerVars_state_geom };
  }
  shape[0]++; // one more timestep
  auto int_offset = tensor_int->size();
  tensor_int->resize(shape);
  auto& tensor_int_vec = tensor_int->get_data();

  // read
  int32_t nlist_offset = dyna_airbag_ngeom + dyna_airbag_state_nvars;
  size_t iFloatVar = 0;
  size_t iIntegerVar = 0;
  for (int32_t ii = 0; ii < wordsToRead; ++ii) {
    if (dyna_airbag_nlist[nlist_offset + ii % dyna_airbag_state_geom] == 1) {
      tensor_int_vec[int_offset + iIntegerVar] =
        this->buffer->read_int(start + ii);
      ++iIntegerVar;
    } else {
      tensor_float_vec[float_offset + iFloatVar] =
        this->buffer->read_float(start + ii);
      ++iFloatVar;
    }
  }

  // PARTICLE STATE DATA

  start += wordsToRead;
  wordsToRead = dyna_airbag_nparticles * dyna_airbag_state_nvars;

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

  // allocate
  const std::string tensor_float2_name = "airbag_particle_float_results";
  std::shared_ptr<Tensor<float>> tensor_float2 = nullptr;
  if (float_data.find(tensor_float2_name) == float_data.end()) {
    tensor_float2 = std::make_shared<Tensor<float>>();
    float_data.insert(std::make_pair(tensor_float2_name, tensor_float2));
  } else {
    tensor_float2 = float_data[tensor_float2_name];
  }
  shape = tensor_float2->get_shape();
  if (shape.size() == 0) {
    shape = {
      0,
      static_cast<size_t>(dyna_airbag_nparticles),
      static_cast<size_t>(nFloatVars_particles),
    };
  }
  shape[0]++; // one more timestep
  float_offset = tensor_float2->size();
  tensor_float2->resize(shape);
  auto& tensor_float_vec2 = tensor_float2->get_data();

  const std::string tensor_int2_name = "airbag_particle_int_results";
  std::shared_ptr<Tensor<int32_t>> tensor_int2 = nullptr;
  if (int_data.find(tensor_int2_name) == int_data.end()) {
    tensor_int2 = std::make_shared<Tensor<int32_t>>();
    int_data.insert(std::make_pair(tensor_int2_name, tensor_int2));
  } else {
    tensor_int2 = int_data[tensor_int2_name];
  }
  shape = tensor_int2->get_shape();
  if (shape.size() == 0) {
    shape = {
      0,
      static_cast<size_t>(dyna_airbag_nparticles),
      static_cast<size_t>(nIntegerVars_particles),
    };
  }
  shape[0]++; // one more timestep
  int_offset = tensor_int2->size();
  tensor_int2->resize(shape);
  auto& tensor_int2_vec = tensor_int2->get_data();

  // read particle results
  iFloatVar = 0;
  iIntegerVar = 0;
  for (int32_t ii = 0; ii < wordsToRead; ++ii) {
    if (dyna_airbag_nlist[dyna_airbag_ngeom + ii % dyna_airbag_state_nvars] ==
        1) {
      tensor_int2_vec[int_offset + iIntegerVar] =
        this->buffer->read_int(start + ii);
      ++iIntegerVar;
    } else {
      tensor_float_vec2[float_offset + iFloatVar] =
        this->buffer->read_float(start + ii);
      ++iFloatVar;
    }
  }
}

/** Checks for the ending mark at a specific word
 *
 * @param iWord : word position to check
 * @return is_ending
 */
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

/** Get string data saved in the file
 *
 * @param _name : name of the variable
 * @return ret : vector with string data
 */
std::vector<std::string>
RawD3plot::get_string_data(const std::string& _name)
{
  auto it = this->string_data.find(_name);
  if (it != this->string_data.end()) {
    return it->second;
  } else {
    throw(std::invalid_argument("Can not find: " + _name));
  }
}

/** Get the names of the string data buffer
 *
 * @return ret : vector of variable names available
 */
std::vector<std::string>
RawD3plot::get_string_names() const
{
  std::vector<std::string> ret;
  for (auto iter : this->string_data) {
    ret.push_back(iter.first);
  }
  return ret;
}

/** Get id data from the file
 *
 * @param _name : variable name
 * @return ret : tensor
 */
std::shared_ptr<Tensor<int32_t>>
RawD3plot::get_int_data(const std::string& _name)
{
  auto it = this->int_data.find(_name);
  if (it != this->int_data.end()) {
    return it->second;
  } else {
    throw(std::invalid_argument("Can not find: " + _name));
  }
}

/** Get id variable names available
 *
 * @param ret : vector of variable names available
 */
std::vector<std::string>
RawD3plot::get_int_names() const
{
  std::vector<std::string> ret;
  for (auto iter : this->int_data) {
    ret.push_back(iter.first);
  }
  return ret;
}

/** Insert an integer memory array into the file buffer
 *
 * @param _name : name of the variable
 * @param _data : data array
 */
void
RawD3plot::set_int_data(const std::string& _name,
                        std::shared_ptr<Tensor<int32_t>> _data)
{
  int_data.insert(std::make_pair(_name, _data));
}

/** Get float data from the file
 *
 * @param _name : variable name
 * @return ret : tensor
 */
std::shared_ptr<Tensor<float>>
RawD3plot::get_float_data(const std::string& _name)
{
  auto it = this->float_data.find(_name);
  if (it != this->float_data.end()) {
    return it->second;
  } else {
    throw(std::invalid_argument("Can not find: " + _name));
  }
}

/** Get float data variables available
 *
 * @param ret : vector of variable names available
 */
std::vector<std::string>
RawD3plot::get_float_names() const
{
  std::vector<std::string> ret;
  for (auto iter : this->float_data) {
    ret.push_back(iter.first);
  }
  return ret;
}

/** Insert an float memory array into the file buffer
 *
 * @param _name : name of the variable
 * @param _shape : shape of the data tensor
 * @param _data_ptr : pointer to the first data element for copy
 */
void
RawD3plot::set_float_data(const std::string& _name,
                          std::vector<size_t> _shape,
                          const float* _data_ptr)
{

  auto tensor = std::make_shared<Tensor<float>>();
  float_data.insert(std::make_pair(_name, tensor));

  if (_shape.size() < 1)
    return;

  tensor->resize(_shape);
  size_t offset = 1;
  for (auto entry : _shape)
    offset *= entry;

  std::copy(_data_ptr, _data_ptr + offset, tensor->get_data().begin());
}

/** Insert an int memory array into the file buffer
 *
 * @param _name : name of the variable
 * @param _shape : shape of the data tensor
 * @param _data_ptr : pointer to the first data element for copy
 */
void
RawD3plot::set_int_data(const std::string& _name,
                        std::vector<size_t> _shape,
                        const int* _data_ptr)
{

  auto tensor = std::make_shared<Tensor<int32_t>>();
  int_data.insert(std::make_pair(_name, tensor));

  if (_shape.size() < 1)
    return;

  tensor->resize(_shape);
  size_t offset = 1;
  for (auto entry : _shape)
    offset *= entry;

  std::copy(_data_ptr, _data_ptr + offset, tensor->get_data().begin());
}

/** Insert a string memory vector into the file buffer
 *
 * @param _name : name of the variable
 * @param _data : data array
 */
void
RawD3plot::set_string_data(const std::string& _name,
                           const std::vector<std::string>& _data)
{

  this->string_data[_name] = _data;
}

/** Get the title of the d3plot
 *
 * @return title
 */
inline std::string
RawD3plot::get_title() const
{
  return this->dyna_title;
}

} // namespace qd