
#include <utility>

#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/HDF5_Utility.hpp>

namespace qd {

/** Open an HDF5 file. If the file already exists, it will be opened in append
 * mode.
 *
 * @param _filepath : path to the file
 * @param _overwrite_run : overwrite the run in the file, not the file itself
 */
H5::H5File
open_hdf5(const std::string& _filepath, bool _overwrite_run)
{

  if (check_ExistanceAndAccess(_filepath)) {
    if (_overwrite_run) {

      // clean up
      delete_file(_filepath);

    } else {
      throw std::invalid_argument("File " + _filepath +
                                  " does already exist. Use overwrite.");
    }
  }

  // open new file
  return H5::H5File(_filepath, H5F_ACC_TRUNC);
}

/*
H5::Group
open_run_folder(H5::H5File& _file,
                const std::string& _run_name,
                bool _overwrite_run)
{

  try {

    // open existing
    std::string channelName = "/" + _run_name;

    if (_file.exists(channelName)) {

      if (_overwrite_run) {

        _file.unlink(channelName);
        // H5Ldelete(m_h5File.getId(), channelName.c_str(), H5P_DEFAULT);

      } else {
        throw std::invalid_argument("A run with name " + _run_name +
                                    "does already exist. Either choose a "
                                    "different name or choose overwrite.");
      }

      // create a new group
    }

    return _file.createGroup("/" + _run_name);

  } catch (H5::GroupIException error) {
    error.printError();
    throw std::runtime_error("Error when opening run group in hdf5.");
  }
}
 */

/**
 *
 */
bool
group_exists(H5::H5File& _file, const std::string& _path)
{

  try {
    _file.openGroup(_path.c_str());
    return true;
  } catch (H5::GroupIException error) {
    return false;
  }
}

/** Check if a path is existing
 *
 * @return is_existing
 */
template<typename T>
void
QD_HDF5::write_vector(const std::string& _path, const std::vector<T> _data)
{
  static_assert(
    std::is_same<T, int32_t>::value || std::is_same<T, int64_t>::value ||
      std::is_same<T, float>::value || std::is_same<T, double>::value,
    "can not write unknown vector type to HDF5 file.");

  if (this->exists(_path)) {
    throw std::invalid_argument("Data vector " + _path +
                                " does already exist.");
  }

  // property list
  int fillvalue = 0; /* Fill value for the dataset */
  DSetCreatPropList plist;
  plist.setFillValue(PredType::NATIVE_INT, &fillvalue);

  // Create dataspace for the dataset in the file.
  hsize_t fdim[] = { FSPACE_DIM1, FSPACE_DIM2 }; // dim sizes of ds (on disk)
  DataSpace fspace(FSPACE_RANK, fdim);

  // dataset
  DataSet* dataset = new DataSet(
    file->createDataSet(DATASET_NAME, PredType::NATIVE_INT, fspace, plist));

  // make constexpr in c++17
  if (std::is_same<T, int32_t>::value) {

  } else if (std::is_same<T, int64_t>::value) {

  } else if (std::is_same<T, float>::value) {

  } else if (std::is_same<T, double>::value) {

  } else {
    throw(
      std::invalid_argument("can not write unknown vector type to HDF5 file."));
  }
}

} // namespace qd