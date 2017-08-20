
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

} // namespace qd