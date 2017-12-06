
#ifndef DB_PARTS_HPP
#define DB_PARTS_HPP

#include <dyna_cpp/db/Part.hpp>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace qd {

// forward declarations
// class Part;
class FEMFile;

class DB_Parts
{
private:
  FEMFile* femfile;
  std::vector<std::shared_ptr<Part>> parts;
  std::unordered_map<int32_t, size_t> id2index_parts;

public:
  explicit DB_Parts(FEMFile* _femfile);
  virtual ~DB_Parts();

  template<typename T>
  size_t get_part_index_from_id(T _id);
  template<typename T>
  T get_part_id_from_index(size_t _index);

  size_t get_nParts() const;
  void print_parts() const;
  std::shared_ptr<Part> add_partByID(int32_t _partID,
                                     const std::string& name = "");
  std::vector<std::shared_ptr<Part>> get_parts();
  std::shared_ptr<Part> get_partByName(const std::string&);
  template<typename T>
  std::shared_ptr<Part> get_partByID(T _id);
  template<typename T>
  std::vector<std::shared_ptr<Part>> get_partByID(std::vector<T> _ids);
  template<typename T>
  std::shared_ptr<Part> get_partByIndex(T _index);
  template<typename T>
  std::vector<std::shared_ptr<Part>> get_partByIndex(std::vector<T> _indexes);
};

template<typename T>
inline size_t
DB_Parts::get_part_index_from_id(T _id)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  const auto& it = this->id2index_parts.find(_id);
  if (it != id2index_parts.end()) {
    return it->second;
  } else {
    // :(
    throw(std::invalid_argument("Could not find part with id " +
                                std::to_string(_id)));
  }
}

template<typename T>
T
DB_Parts::get_part_id_from_index(size_t _index)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  return this->get_partByIndex(_index)->get_partID();
}

/** Get a part from a single id
 * @param _id
 * @return part_ptr
 */
template<typename T>
std::shared_ptr<Part>
DB_Parts::get_partByID(T _id)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  return this->get_partByIndex(this->get_part_index_from_id(_id));
}

/** Get a part from a list of ids
 * @param _ids
 * @return vector of parts
 */
template<typename T>
std::vector<std::shared_ptr<Part>>
DB_Parts::get_partByID(std::vector<T> _ids)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  std::vector<std::shared_ptr<Part>> ret;
  for (auto& id : _ids) {
    ret.push_back(this->get_partByID(id));
  }

  return std::move(ret);
}

/** Get a part from an internal index
 * @param _index
 * @return part_ptr
 *
 * The index must be smaller than the number of parts.
 */
template<typename T>
inline std::shared_ptr<Part>
DB_Parts::get_partByIndex(T _index)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  try {
    return this->parts.at(_index);
  } catch (const std::out_of_range&) {
    throw(std::invalid_argument("Could not find part with index " +
                                std::to_string(_index)));
  }
}

/** Get a part from a list of indexes
 * @param _indexes
 * @return vector of parts
 */
template<typename T>
std::vector<std::shared_ptr<Part>>
DB_Parts::get_partByIndex(std::vector<T> _indexes)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  std::vector<std::shared_ptr<Part>> ret;
  for (auto& index : _indexes) {
    ret.push_back(this->get_partByIndex(index));
  }

  return std::move(ret);
}

} // namespace qd

#endif
