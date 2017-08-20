
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

} // namespace qd

#endif