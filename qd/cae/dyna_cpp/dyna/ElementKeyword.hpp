#ifndef ELEMENTKEYWORD_HPP
#define ELEMENTKEYWORD_HPP

#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/dyna/Keyword.hpp>

namespace qd {

class ElementKeyword : public Keyword
{
private:
  DB_Elements* db_elems;
  Element::ElementType type;
  std::vector<size_t> elem_indexes_in_card;
  std::vector<int32_t> elem_part_ids;
  std::vector<std::string> unparsed_element_data;
  std::vector<std::string> trailing_lines;

  Element::ElementType determine_elementType(
    const std::string& _keyword_name) const;
  void parse_elem2(const std::string& _keyword_name_lower,
                   const std::vector<std::string>& _lines);

public:
  explicit ElementKeyword(DB_Elements* _db_elems,
                          const std::string& _lines,
                          int64_t _iLine = 0);
  inline Element::ElementType get_type() const;
  inline size_t get_nElements() const;
  template<typename T>
  std::shared_ptr<Element> get_elementByIndex(T _index);

  template<typename T>
  std::shared_ptr<Element>
  add_elementByNodeID(T _id, T _part_id, const std::vector<int32_t>& _node_ids);
  template<typename T>
  std::shared_ptr<Element> add_elementByNodeIndex(
    T _id,
    T _part_id,
    const std::vector<int32_t>& _node_indexes);
  std::vector<std::shared_ptr<Element>> get_elements();
  std::string str() override;
};

/** Get the element type of the keyword
 *
 * @return type : element type
 */
Element::ElementType
ElementKeyword::get_type() const
{
  return type;
}

/** Get the number of elements in the keyword
 *
 * @param _type : optional element type specify
 * @return nElements
 */
size_t
ElementKeyword::get_nElements() const
{
  return elem_indexes_in_card.size();
}

/** Get an element in the card from it's index
 *
 * @param _type : element type
 * @param _index : index of the element
 * @return element
 */
template<typename T>
std::shared_ptr<Element>
ElementKeyword::get_elementByIndex(T _index)
{
  _index = index_treatment(_index, elem_indexes_in_card.size());
  return db_elems->get_elementByIndex(type, elem_indexes_in_card[_index]);
}

/** Add an element by the indices of a node
 *
 * @param _type : type of the element
 * @param _id : id of the element
 * @param _part_id : id of the part the element belongs to
 * @param _node_indexes : indexes of nodes
 */
template<typename T>
std::shared_ptr<Element>
ElementKeyword::add_elementByNodeIndex(
  T _id,
  T _part_id,
  const std::vector<int32_t>& _node_indexes)
{
  auto elem = db_elems->add_elementByNodeIndex(
    type, static_cast<int32_t>(_id), _part_id, _node_indexes);
  elem_indexes_in_card.push_back(
    db_elems->get_element_index_from_id(type, _id));
  return elem;
}

/** Add an element by the indices of a node
 *
 * @param _type : type of the element
 * @param _id : id of the element
 * @param _part_id : id of the part the element belongs to
 * @param _node_indexes : indexes of nodes
 */
template<typename T>
std::shared_ptr<Element>
ElementKeyword::add_elementByNodeID(T _id,
                                    T _part_id,
                                    const std::vector<int32_t>& _node_ids)
{
  auto elem = db_elems->add_elementByNodeID(
    type, static_cast<int32_t>(_id), _part_id, _node_ids);
  elem_indexes_in_card.push_back(
    db_elems->get_element_index_from_id(type, _id));
  return elem;
}

} // namespace:qd

#endif