
#ifndef HDF5_UTILITY_HPP
#define HDF5_UTILITY_HPP

#include <string>

#include <H5Cpp.h>

namespace qd {

H5::H5File
open_hdf5(const std::string& _filepath, bool _overwrite_run);

H5::Group
open_run_folder(H5::H5File& _file,
                const std::string& _run_name,
                bool _overwrite_run);

/** QD wrapper for hdf5 files.
 *
 * Same as H5File but with increased utility.
 */
class QD_HDF5 : public H5::H5File
{

public:
  using H5::H5File::H5File; // inherit constructor
  template<typename T>
  void write_vector(const std::string& _path, const std::vector<T> _data);
};

} // namespace qd

#endif