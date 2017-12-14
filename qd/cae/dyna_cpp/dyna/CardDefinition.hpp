
#ifndef CARDDEFINITION_HPP
#define CARDDEFINITION_HPP

// includes
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <dyna_cpp/dyna/CardEntryDefinition.hpp>
#include <dyna_cpp/dyna/KeywordDefinition.hpp>

namespace qd {

class CardDefinition
{
private:
  bool is_optional;
  std::shared_ptr<KeywordDefinition> parent_keyword;
  std::vector<std::shared_ptr<CardEntryDefinition>> card_entry_defs;

public:
  CardDefinition(std::shared_ptr<KeywordDefinition> _parent, bool _is_optional);
  void add_card_entry_definition(std::shared_ptr<CardEntryDefinition> _entry);

  std::shared_ptr<CardEntryDefinition> get_card_entry_definition(
    std::string _name);
  std::shared_ptr<CardEntryDefinition> get_card_entry_definition(
    int64_t _index);
};

} // namespace qd

#endif