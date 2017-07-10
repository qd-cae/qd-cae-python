
#ifndef DB_ELEMENTS_HPP
#define DB_ELEMENTS_HPP

// forward declarations
// class Element;
class DB_Nodes;
class DB_Parts;
class FEMFile;

// includes
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

class DB_Elements
{
private:
  FEMFile* femfile;
  DB_Nodes* db_nodes;
  DB_Parts* db_parts;

  std::unordered_map<int, size_t> id2index_elements2;
  std::unordered_map<int, size_t> id2index_elements4;
  std::unordered_map<int, size_t> id2index_elements8;
  std::vector<std::shared_ptr<Element>> elements2;
  std::vector<std::shared_ptr<Element>> elements4;
  std::vector<std::shared_ptr<Element>> elements8;

public:
  DB_Elements(FEMFile* _femfile);
  virtual ~DB_Elements();
  FEMFile* get_femfile();
  DB_Nodes* get_db_nodes();

  // memory stuff
  void reserve(const Element::ElementType _type, const size_t _size);
  std::shared_ptr<Element> add_element_byD3plot(
    const Element::ElementType _eType,
    const int _id,
    const std::vector<int>& _elem_data);
  std::shared_ptr<Element> add_element_byKeyFile(Element::ElementType _eType,
                                                 int _id,
                                                 int _partid,
                                                 std::vector<int> _node_ids);

  // getter
  size_t get_nElements(const Element::ElementType _type = Element::NONE) const;
  std::vector<std::shared_ptr<Element>> get_elements(
    const Element::ElementType _type = Element::NONE);
  template<typename T>
  std::shared_ptr<Element> get_elementByID(Element::ElementType _eType, T _id);
  template<typename T>
  std::shared_ptr<Element> get_elementByIndex(Element::ElementType _eType,
                                              T _index);
  template<typename T>
  std::vector<std::shared_ptr<Element>> get_elementByID(
    Element::ElementType _eType,
    const std::vector<T>& _ids);
  template<typename T>
  std::vector<std::shared_ptr<Element>> get_elementByIndex(
    Element::ElementType _eType,
    const std::vector<T>& _indexes);

  // Python API
  template<typename T>
  std::vector<std::shared_ptr<Element>> get_elementByID(
    Element::ElementType _eType,
    pybind11::list _list)
  {
    return this->get_elementByID(
      _eType,
      qd::py::container_to_vector<T>(
        _list, "An entry of the id list was not an integer."));
  };
  template<typename T>
  std::vector<std::shared_ptr<Element>> get_elementByID(
    Element::ElementType _eType,
    pybind11::tuple _tuple)
  {
    return this->get_elementByID(
      _eType,
      qd::py::container_to_vector<T>(
        _tuple, "An entry of the id list was not an integer."));
  };
  template<typename T>
  std::vector<std::shared_ptr<Element>> get_elementByIndex(
    Element::ElementType _eType,
    pybind11::list _list)
  {
    return this->get_elementByIndex(
      _eType,
      qd::py::container_to_vector<T>(
        _list, "An entry of the index list was not an integer."));
  };
  template<typename T>
  std::vector<std::shared_ptr<Element>> get_elementByIndex(
    Element::ElementType _eType,
    pybind11::tuple _tuple)
  {
    return this->get_elementByIndex(
      _eType,
      qd::py::container_to_vector<T>(
        _tuple, "An entry of the index list was not an integer."));
  };
};

/** Get the element by it's internal id and it's type.
 *
 * @param _elementType : see description
 * @param _elementID : id of the element (like in the solver)
 * @return _element
 *
 * Type may be: Element.ElementType
 * NONE = 0 -> error
 * BEAM = 1
 * SHELL = 2
 * SOLID = 3
 */
template<typename T>
std::shared_ptr<Element>
DB_Elements::get_elementByID(Element::ElementType _elementType, T _elementID)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  if (_elementType == Element::ElementType::BEAM) {
    const auto& it = this->id2index_elements2.find(_elementID);
    if (it == id2index_elements2.end())
      throw(std::invalid_argument("Can not find beam element with id " +
                                  to_string(_elementID) + " in database"));
    return elements2[it->second];

  } else if (_elementType == Element::ElementType::SHELL) {
    const auto& it = this->id2index_elements4.find(_elementID);
    if (it == id2index_elements4.end())
      throw(std::invalid_argument("Can not find shell element with id " +
                                  to_string(_elementID) + " in database"));
    return elements4[it->second];

  } else if (_elementType == Element::ElementType::SOLID) {
    const auto& it = this->id2index_elements8.find(_elementID);
    if (it == id2index_elements8.end())
      throw(std::invalid_argument("Can not find solid element with id " +
                                  to_string(_elementID) + " in database"));
    return elements8[it->second];
  }

  throw(std::invalid_argument("Can not get element with elementType:" +
                              to_string(_elementID)));
}

/** Get the element by a list of ids and it's type.
 *
 * @param _elementType : see description
 * @param _ids : vector of ids
 * @return ret vector of elements
 *
 * Type may be: Element.ElementType
 * NONE = 0 -> error
 * BEAM = 1
 * SHELL = 2
 * SOLID = 3
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
  return ret;
}

/** Get the element by it's internal index and it's type.
 *
 * @param int _elementType : see description
 * @param int _elementIndex : index of the element (not id!)
 * @return std::shared_ptr<Element> _element
 *
 * Type may be: Element.ElementType
 * NONE = 0 -> error
 * BEAM = 1
 * SHELL = 2
 * SOLID = 3
 */
template<typename T>
// typename std::enable_if<std::is_integral<T>::value>::type
std::shared_ptr<Element>
DB_Elements::get_elementByIndex(Element::ElementType _elementType,
                                T _elementIndex)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  if (_elementType == Element::ElementType::BEAM) {
    if (_elementIndex < elements2.size()) {
      return elements2[_elementIndex];
    } else {
      throw(std::invalid_argument("beam element index " +
                                  to_string(_elementIndex) +
                                  " exceeds the number of elements."));
    }

  } else if (_elementType == Element::ElementType::SHELL) {
    if (_elementIndex < elements4.size()) {
      return elements4[_elementIndex];
    } else {
      throw(std::invalid_argument("shell element index " +
                                  to_string(_elementIndex) +
                                  " exceeds the number of elements."));
    }

  } else if (_elementType == Element::ElementType::SOLID) {
    if (_elementIndex < elements8.size()) {
      return elements8[_elementIndex];
    } else {
      throw(std::invalid_argument("solid element index " +
                                  to_string(_elementIndex) +
                                  " exceeds the number of elements."));
    }
  }

  throw(std::invalid_argument("Can not get element with elementType " +
                              to_string(_elementIndex)));
}

/** Get the element by a list of internal indexes and it's type.
 *
 * @param _elementType : type of the element
 * @param _indexes : vector of indexes
 * @return ret : vector of elements
 *
 * Type may be: Element.ElementType
 * NONE = 0 -> error
 * BEAM = 1
 * SHELL = 2
 * SOLID = 3
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
  return ret;
}

#endif
