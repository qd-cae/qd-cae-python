
#ifndef DB_ELEMENTS_HPP
#define DB_ELEMENTS_HPP

// includes
#include <cstdint>
#include <memory>
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
  std::shared_ptr<Element> DB_Elements::add_elementByNodeID(
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

  template<typename T>
  void delete_elementByIndex(Element::ElementType _eType,
                             const std::vector<T>& _elem_indexes);
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
// typename std::enable_if<std::is_integral<T>::value>::type
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

/** Remove elements from an index list
 *
 * @param _eType : type of the element to remove
 * @param _indexes : indexes of the elements to remove
 *
 * invalid indexes will be skipped
 */
template<typename T>
void
DB_Elements::delete_elementByIndex(Element::ElementType _eType,
                                   const std::vector<T>& _elem_indexes)
{

  if (_eType == Element::NONE)
    throw(std::invalid_argument("Invalid element type for deletion."));

  if (_elem_indexes.empty())
    return;

  // first sort from bigger to less and make unique!
  std::set<T> indexes_unique(_elem_indexes.begin(), _elem_indexes.end());
  std::vector<T> indexes_sorted(indexes_unique.rbegin(), indexes_unique.rend());

  // prepare copying
  auto nElements = get_nElements(_eType);

  std::vector<std::shared_ptr<Element>> new_elements;
  std::unordered_map<int32_t, size_t> new_id2index;

  new_elements.reserve(nElements - indexes_sorted.size());
  new_id2index.reserve(nElements - indexes_sorted.size());

  // do the thing
  for (size_t iElement = 0; iElement < nElements; ++iElement) {

    auto element = get_elementByIndex(_eType, iElement);

    // delete elements
    if (iElement == indexes_sorted.back()) {
      for (auto node : element->get_nodes())
        node->remove_element(element);
      db_parts->get_partByID(element->get_part_id())->remove_element(element);
      indexes_sorted.pop_back();
      continue;
    }

    new_elements.push_back(element);
    new_id2index[element->get_elementID()] = iElement;
  }

  // assign new elements
  switch (_eType) {
    case (Element::BEAM):
      elements2 = std::move(new_elements);
      id2index_elements2 = std::move(new_id2index);
      break;
    case (Element::SHELL):
      elements4 = std::move(new_elements);
      id2index_elements4 = std::move(new_id2index);
      break;
    case (Element::SOLID):
      elements8 = std::move(new_elements);
      id2index_elements8 = std::move(new_id2index);
      break;
    case (Element::TSHELL):
      elements4th = std::move(new_elements);
      id2index_elements4th = std::move(new_id2index);
      break;
  }
}

} // namespace qd

#endif
