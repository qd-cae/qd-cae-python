
#ifndef BINOUT_HPP
#define BINOUT_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

extern "C"
{
#include <dyna_cpp/dyna/binout/lsda/lsda.h>
}

#include <dyna_cpp/math/Tensor.hpp>

namespace qd {

class Binout
{
public:
  enum class EntryType
  {
    UNKNOWN,
    DIRECTORY = 0,
    INT8 = 1,
    INT16 = 2,
    INT32 = 3,
    INT64 = 4,
    UINT8 = 5,
    UINT16 = 6,
    UINT32 = 7,
    UINT64 = 8,
    FLOAT32 = 9,
    FLOAT64 = 10,
    LINK = 11
  };

private:
  int fhandle;
  std::string filepath;

  template<typename T>
  bool _type_is_same(EntryType entry_type);

public:
  Binout(const std::string& filepath);
  ~Binout();
  void cd(const std::string& _path);
  bool exists(const std::string& _path);
  bool has_children(const std::string& _path);
  bool is_variable(const std::string& _path);
  EntryType get_type_id(std::string path);
  std::vector<std::string> get_children(const std::string& _folder_name = ".");
  template<typename T>
  Tensor_ptr<T> read_variable(const std::string& _path);
};

/** Check if LSDA type matches the C++ type
 *
 *  @param entry_type
 */
template<typename T>
bool
Binout::_type_is_same(EntryType entry_type)
{
  if (entry_type == EntryType::INT8 && typeid(T) == typeid(int8_t))
    return true;
  else if (entry_type == EntryType::INT16 && typeid(T) == typeid(int16_t))
    return true;
  else if (entry_type == EntryType::INT32 && typeid(T) == typeid(int32_t))
    return true;
  else if (entry_type == EntryType::INT64 && typeid(T) == typeid(int64_t))
    return true;
  else if (entry_type == EntryType::UINT8 && typeid(T) == typeid(uint8_t))
    return true;
  else if (entry_type == EntryType::UINT16 && typeid(T) == typeid(uint16_t))
    return true;
  else if (entry_type == EntryType::UINT32 && typeid(T) == typeid(uint32_t))
    return true;
  else if (entry_type == EntryType::UINT64 && typeid(T) == typeid(uint64_t))
    return true;
  else if (entry_type == EntryType::FLOAT32 && typeid(T) == typeid(float))
    return true;
  else if (entry_type == EntryType::FLOAT64 && typeid(T) == typeid(double))
    return true;
  else
    return false;
}

/** Read the variable data
 *
 * @param path : path to the variable
 * @return tensor_ptr : tensor instance
 */
template<typename T>
Tensor_ptr<T>
Binout::read_variable(const std::string& path)
{

  if (!this->exists(path))
    throw(std::invalid_argument("Path " + path + " does not exist."));

  if (!this->is_variable(path))
    throw(
      std::invalid_argument("Path " + path + " does not point to a variable."));

  auto entry_type = this->get_type_id(path);
  if (entry_type == EntryType::UNKNOWN)
    throw(std::runtime_error("Path " + path +
                             " caused an error in Binout.read_variable. This "
                             "should never happen ..."));
  else if (entry_type == EntryType::DIRECTORY)
    throw(std::invalid_argument(
      "Path " + path + " does point to a DIRECTORY and not to a variable."));
  else if (entry_type == EntryType::LINK)
    throw(std::invalid_argument(
      "Path " + path + " does point to a LINK and not to a variable."));
  else if (!this->_type_is_same<T>(entry_type))
    throw(std::runtime_error(
      "C++ Tensor type does not match Binout variable type."));

  size_t length = 0;
  int32_t type_id = -1;
  int32_t filenum = -1;
  lsda_queryvar(this->fhandle, (char*)&path[0], &type_id, &length, &filenum);
  if (type_id < 0)
    throw(std::invalid_argument(
      "Binout.read_variable encountered an error on: " + path));

  auto tensor_ptr = std::make_shared<Tensor<T>>();
  tensor_ptr->resize({ length });
  auto& data = tensor_ptr->get_data();

  size_t ret = lsda_lread(this->fhandle,
                          static_cast<int32_t>(type_id),
                          (char*)&path[0],  // path
                          0,                // offset
                          length,           // length
                          (void*)&data[0]); // data

  // I know this is dumb, but dyna defines an unsigned type with -1 in case of
  // an error. Really need to fix this sometime ...
  if (ret == (size_t)-1) {
    throw(std::runtime_error(
      "Error during lsda_realread and I have no clue what happened. Sry."));
  }

  return tensor_ptr;
}

} // namespace qd

#endif // Binout
