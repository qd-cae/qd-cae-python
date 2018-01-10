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
  std::vector<size_t> elem2_indexes_in_card;
  std::vector<size_t> elem4_indexes_in_card;
  std::vector<size_t> elem8_indexes_in_card;

  std::vector<std::string> trailing_lines;

public:
  explicit ElementKeyword(DB_Elements* db_elems,
                          const std::string& _lines,
                          int64_t _iLine = 0);
  inline size_t get_nElements() const;
  template<typename T>
  std::shared_ptr<Element> get_elementByIndex(Element::ElementType _type,
                                              T _index);

  template<typename T>
  std::shared_ptr<Element> add_elementByID(Element::ElementType _type,
                                           T _id,
                                           std::vector<int32_t> _node_ids);
  template<typename T>
  std::shared_ptr<Element> add_elementByIndex(
    Element::ElementType _type,
    T _id,
    std::vector<int32_t> _node_indexes);
  std::vector<std::shared_ptr<Element>> get_elements();
  std::string str() override;
};

} // namespace:qd

#endif