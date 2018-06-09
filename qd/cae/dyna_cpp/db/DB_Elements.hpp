
#ifndef DB_ELEMENTS_HPP
#define DB_ELEMENTS_HPP

// includes
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/db/Part.hpp>

namespace qd {

// forward declarations
class DB_Nodes;
class DB_Parts;
class FEMFile;

class DB_Elements
{
private:
  std::mutex _elem2_mutex;
  std::mutex _elem4_mutex;
  std::mutex _elem4th_mutex;
  std::mutex _elem8_mutex;

  FEMFile* femfile;
  DB_Nodes* db_nodes;
  DB_Parts* db_parts;

  std::unordered_map<int32_t, size_t> id2index_elements2;
  std::unordered_map<int32_t, size_t> id2index_elements4;
  std::unordered_map<int32_t, size_t> id2index_elements4th;
  std::unordered_map<int32_t, size_t> id2index_elements8;
  std::vector<std::shared_ptr<Element>> elements2;
  std::vector<std::shared_ptr<Element>> elements4;
  std::vector<std::shared_ptr<Element>> elements4th;
  std::vector<std::shared_ptr<Element>> elements8;

  std::shared_ptr<Element> create_element_unchecked(
    Element::ElementType _eType,
    int32_t _element_id,
    int32_t _part_id,
    const std::vector<int32_t>& _node_ids);

public:
  explicit DB_Elements(FEMFile* _femfile);
  virtual ~DB_Elements();
  FEMFile* get_femfile();
  DB_Nodes* get_db_nodes();

  // memory stuff
  void reserve(const Element::ElementType _type, const size_t _size);
  std::shared_ptr<Element> add_elementByNodeIndex(
    const Element::ElementType _eType,
    int32_t _id,
    int32_t _part_id,
    const std::vector<size_t>& _node_indexes);
  std::shared_ptr<Element> add_elementByNodeID(
    const Element::ElementType _eType,
    int32_t _elementID,
    int32_t _part_id,
    const std::vector<int32_t>& _node_ids);
  std::shared_ptr<Element> add_element_byD3plot(
    const Element::ElementType _eType,
    const int32_t _id,
    const std::vector<int32_t>& _elem_data);

  // getter
  size_t get_nElements(const Element::ElementType _type = Element::NONE) const;
  std::vector<std::shared_ptr<Element>> get_elements(
    const Element::ElementType _type = Element::NONE);
  template<typename T>
  T get_element_id_from_index(Element::ElementType _type, size_t _index);
  template<typename T>
  size_t get_element_index_from_id(Element::ElementType _type, T _id);

  template<typename T>
  std::shared_ptr<Element> get_elementByID(Element::ElementType _eType, T _id);
  template<typename T>
  std::vector<std::shared_ptr<Element>> get_elementByID(
    Element::ElementType _eType,
    const std::vector<T>& _ids);
  template<typename T>
  std::shared_ptr<Element> get_elementByIndex(Element::ElementType _eType,
                                              T _index);
  template<typename T>
  std::vector<std::shared_ptr<Element>> get_elementByIndex(
    Element::ElementType _eType,
    const std::vector<T>& _indexes);

  // array functions
  Tensor_ptr<float> get_element_energy(
    Element::ElementType element_filter = Element::ElementType::NONE);
  Tensor_ptr<float> get_element_stress_mises(
    Element::ElementType element_filter = Element::ElementType::NONE);
  Tensor_ptr<float> get_element_plastic_strain(
    Element::ElementType element_filter = Element::ElementType::NONE);
  Tensor_ptr<float> get_element_strain(
    Element::ElementType element_filter = Element::ElementType::NONE);
  Tensor_ptr<float> get_element_stress(
    Element::ElementType element_filter = Element::ElementType::NONE);
  Tensor_ptr<float> get_element_coords(
    Element::ElementType element_filter = Element::ElementType::NONE);
  Tensor_ptr<float> get_element_history_vars(Element::ElementType element_type);

  Tensor_ptr<int32_t> get_element_ids(
    Element::ElementType element_filter = Element::ElementType::NONE);
  Tensor_ptr<int32_t> get_element_node_ids(Element::ElementType element_type,
                                           size_t n_nodes);
};

/** Get the element idnex from an id
 *
 * @param _id : element id
 * @return _index : index of the element
 */
template<typename T>
size_t
DB_Elements::get_element_index_from_id(Element::ElementType _type, T _id)
{

  switch (_type) {

    case Element::ElementType::BEAM: {
      const auto& it = this->id2index_elements2.find(_id);
      if (it == id2index_elements2.end())
        throw(std::invalid_argument("Can not find beam element with id " +
                                    std::to_string(_id) + " in database"));
      return it->second;
      break;
    }

    case Element::ElementType::SHELL: {
      const auto& it = this->id2index_elements4.find(_id);
      if (it == id2index_elements4.end())
        throw(std::invalid_argument("Can not find shell element with id " +
                                    std::to_string(_id) + " in database"));
      return it->second;
      break;
    }

    case Element::ElementType::SOLID: {
      const auto& it = this->id2index_elements8.find(_id);
      if (it == id2index_elements8.end())
        throw(std::invalid_argument("Can not find solid element with id " +
                                    std::to_string(_id) + " in database"));
      return it->second;
      break;
    }

    case Element::ElementType::TSHELL: {
      const auto& it = this->id2index_elements4th.find(_id);
      if (it == id2index_elements4th.end())
        throw(
          std::invalid_argument("Can not find thick shell element with id " +
                                std::to_string(_id) + " in database"));
      return it->second;
      break;
    }

    default:
      throw(std::invalid_argument("Can not get element with type:" +
                                  std::to_string(_type)));
      break;
  }
}

/** Get the element id from an index
 *
 * @param _index : element index
 * @return _id : id of the element
 */
template<typename T>
T
DB_Elements::get_element_id_from_index(Element::ElementType _type,
                                       size_t _index)
{
  return this->get_elementByIndex(_type, _index)->get_elementID();
}

/** Get the element by it's internal id and it's type.
 *
 * @param _elementType : see description
 * @param _elementID : id of the element (like in the solver)
 * @return _element
 *
 */
template<typename T>
std::shared_ptr<Element>
DB_Elements::get_elementByID(Element::ElementType _type, T _id)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  return this->get_elementByIndex(_type,
                                  this->get_element_index_from_id(_type, _id));
}

/** Get the element by a list of ids and it's type.
 *
 * @param _elementType : see description
 * @param _ids : vector of ids
 * @return ret vector of elements
 *
 */
template<typename T>
std::vector<std::shared_ptr<Element>>
DB_Elements::get_elementByID(Element::ElementType _elementType,
                             const std::vector<T>& _ids)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  std::vector<std::shared_ptr<Element>> ret;
  for (const auto id : _ids) {
    ret.push_back(this->get_elementByID(_elementType, id));
  }
  return std::move(ret);
}

/** Get the element by it's internal index and it's type.
 *
 * @param int32_t _elementType : see description
 * @param int32_t _elementIndex : index of the element (not id!)
 * @return std::shared_ptr<Element> _element
 *
 */
template<typename T>
std::shared_ptr<Element>
DB_Elements::get_elementByIndex(Element::ElementType _type, T _index)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  switch (_type) {

    case Element::ElementType::BEAM: {
      try {
        return this->elements2.at(_index);
      } catch (const std::out_of_range&) {
        throw(std::invalid_argument("Could not find beam element with index " +
                                    std::to_string(_index)));
      }
      break;
    }

    case Element::ElementType::SHELL: {
      try {
        return this->elements4.at(_index);
      } catch (const std::out_of_range&) {
        throw(std::invalid_argument("Could not find shell element with index " +
                                    std::to_string(_index)));
      }
      break;
    }

    case Element::ElementType::SOLID: {
      try {
        return this->elements8.at(_index);
      } catch (const std::out_of_range&) {
        throw(std::invalid_argument("Could not find solid element with index " +
                                    std::to_string(_index)));
      }
      break;
    }

    case Element::ElementType::TSHELL: {
      try {
        return this->elements4th.at(_index);
      } catch (const std::out_of_range&) {
        throw(std::invalid_argument("Could not find solid element with index " +
                                    std::to_string(_index)));
      }
      break;
    }

    default:
      throw(std::invalid_argument("Can not get element with type:" +
                                  std::to_string(_type)));
      break;
  }
}

/** Get the element by a list of internal indexes and it's type.
 *
 * @param _elementType : type of the element
 * @param _indexes : vector of indexes
 * @return ret : vector of elements
 *
 */
template<typename T>
std::vector<std::shared_ptr<Element>>
DB_Elements::get_elementByIndex(Element::ElementType _elementType,
                                const std::vector<T>& _indexes)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  std::vector<std::shared_ptr<Element>> ret;
  for (const auto index : _indexes) {
    ret.push_back(this->get_elementByIndex(_elementType, index));
  }
  return std::move(ret);
}

} // namespace qd

#endif
