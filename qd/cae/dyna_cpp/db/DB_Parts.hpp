
#ifndef DB_PARTS_HPP
#define DB_PARTS_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

// forward declarations
class Part;
class FEMFile;

class DB_Parts {
 private:
  FEMFile* femfile;
  std::vector<std::shared_ptr<Part>> parts;
  std::unordered_map<int, size_t> id2index_parts;

 public:
  DB_Parts(FEMFile* _femfile);
  virtual ~DB_Parts();

  size_t get_nParts() const;
  void print_parts() const;
  std::shared_ptr<Part> add_partByID(int _partID, const std::string& name = "");

  std::vector<std::shared_ptr<Part>> get_parts();
  std::shared_ptr<Part> get_partByName(const std::string&);
  template <typename T>
  std::shared_ptr<Part> get_partByID(T _id);
  template <typename T>
  std::shared_ptr<Part> get_partByIndex(T _index);
};

template <typename T>
std::shared_ptr<Part> DB_Parts::get_partByID(T _id) {
  static_assert(std::is_integral<T>::value, "Integer number required.");

  const auto& it = this->id2index_parts.find(_id);
  if (it != id2index_parts.end()) {
    return parts[it->second];
  } else {
    // :(
    throw(
        std::invalid_argument("Could not find part with id " + to_string(_id)));
    return nullptr;
  }
}

template <typename T>
std::shared_ptr<Part> DB_Parts::get_partByIndex(T _index) {
  static_assert(std::is_integral<T>::value, "Integer number required.");

  // Part existing
  if (_index < parts.size()) {
    return parts[_index];
  } else {
    // :(
    throw(std::invalid_argument("Could not find part with index " +
                                to_string(_index)));
    return nullptr;
  }
}
#endif
