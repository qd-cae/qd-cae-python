
#ifndef BINOUT_HPP
#define BINOUT_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <dyna_cpp/dyna/binout/lsda/lsda.h>

namespace qd {

class Binout
{

private:
  int fhandle;
  std::string filepath;

public:
  Binout(const std::string& filepath);
  ~Binout();
  void cd(const std::string& _path);
  bool exists(const std::string& _path);
  bool has_children(const std::string& _path);
  bool is_variable(const std::string& _path);
  std::vector<std::string> get_children(const std::string& _folder_name = "");
  template<typename T>
  std::vector<T> read_variable(const std::string& _path);
};

/** Read a vector of data
 * @param _path
 * @return _data
 */
template<typename T>
std::vector<T>
Binout::read_variable(const std::string& _path)
{

  size_t length = 0;
  int32_t type_id = -1;
  int32_t filenum = -1;
  lsda_queryvar(this->fhandle, &_path[0], &type_id, &length, &filenum);
  if (type_id < 0)
    throw(std::invalid_argument(
      "Binout.read_variable encountered an error on: " + _path));

  std::vector<T> data(length);
  size_t ret = lsda_lread(this->fhandle,
                          type_id,
                          &_path[0],        // path
                          0,                // offset
                          length,           // length
                          (void*)&data[0]); // data

  // I know this is dumb, but dyna defines an unsigned type with -1 in case of
  // an error. Really need to fix this sometime ...
  if (ret == (size_t)-1) {
    throw(std::runtime_error("Error during lsda_realread."));
  }

  return data;
}

} // namespace qd

#endif // Binout
