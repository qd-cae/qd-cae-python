#ifndef ELEMENTKEYWORD_HPP
#define ELEMENTKEYWORD_HPP

#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/dyna/keyfile/Keyword.hpp>

namespace qd {

class ElementKeyword : public Keyword
{
private:
  DB_Elements* db_elems;
  Element::ElementType element_type;
  std::vector<size_t> elem_indexes_in_card;
  std::vector<int32_t> elem_part_ids;
  std::vector<std::string> unparsed_element_data;
  std::vector<std::string> trailing_lines;

  Element::ElementType determine_element_type(
    const std::string& _keyword_name) const;
  void parse_elem2(const std::string& _keyword_name_lower,
                   const std::vector<std::string>& _lines);
  void parse_elem4(const std::string& _keyword_name_lower,
                   const std::vector<std::string>& _lines);
  void parse_elem8(const std::string& _keyword_name_lower,
                   const std::vector<std::string>& _lines);
  void parse_elem4th(const std::string& _keyword_name_lower,
                     const std::vector<std::string>& _lines);
  void keyword_elem2_str(std::stringstream& _ss);
  void keyword_elem4_str(std::stringstream& _ss);
  void keyword_elem8_str(std::stringstream& _ss);
  void keyword_elem4th_str(std::stringstream& _ss);

public:
  explicit ElementKeyword(DB_Elements* _db_elems,
                          const std::vector<std::string>& _lines,
                          int64_t _iLine = 0);
  void load();

  inline Element::ElementType get_element_type() const;
  inline size_t get_nElements() const;
  template<typename T>
  std::shared_ptr<Element> get_elementByIndex(T _index);

  template<typename T>
  std::shared_ptr<Element> add_elementByNodeID(
    T _id,
    T _part_id,
    const std::vector<int32_t>& _node_ids,
    const std::vector<std::string>& _additional_card_data = "");
  template<typename T>
  std::shared_ptr<Element> add_elementByNodeIndex(
    T _id,
    T _part_id,
    const std::vector<size_t>& _node_indexes,
    const std::vector<std::string>& _additional_card_data = "");
  std::vector<std::shared_ptr<Element>> get_elements();
  std::string str() override;
};

/** Get the element type of the keyword
 *
 * @return type : element type
 */
Element::ElementType
ElementKeyword::get_element_type() const
{
  return element_type;
}

/** Get the number of elements in the keyword
 *
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
  static_assert(std::is_integral<T>::value, "Integer number required.");

  _index = index_treatment(_index, elem_indexes_in_card.size());
  return db_elems->get_elementByIndex(element_type,
                                      elem_indexes_in_card[_index]);
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
  const std::vector<size_t>& _node_indexes,
  const std::vector<std::string>& _additional_card_data)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  // create element
  auto id = static_cast<int32_t>(_id);
  auto elem = db_elems->add_elementByNodeIndex(
    element_type, id, static_cast<int32_t>(_part_id), _node_indexes);

  // save additional info
  elem_indexes_in_card.push_back(
    db_elems->get_element_index_from_id(element_type, id));
  elem_part_ids.push_back(static_cast<int32_t>(_part_id));
  unparsed_element_data.push_back(str_concat_lines(_additional_card_data));

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
ElementKeyword::add_elementByNodeID(
  T _id,
  T _part_id,
  const std::vector<int32_t>& _node_ids,
  const std::vector<std::string>& _additional_card_data)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  // create element
  auto id = static_cast<int32_t>(_id);
  auto elem = db_elems->add_elementByNodeID(
    element_type, id, static_cast<int32_t>(_part_id), _node_ids);

  // save additional info
  elem_indexes_in_card.push_back(
    db_elems->get_element_index_from_id(element_type, id));
  elem_part_ids.push_back(static_cast<int32_t>(_part_id));
  unparsed_element_data.push_back(str_concat_lines(_additional_card_data));

  return elem;
}

} // namespace:qd

#endif