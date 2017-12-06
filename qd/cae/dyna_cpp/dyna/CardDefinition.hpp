
#ifndef CARDDEFINITION_HPP
#define CARDDEFINITION_HPP

// includes
#include <cstdint>
#include <string>
#include <vector>

#include <dyna_cpp/dyna/CardEntryDefinition.hpp>

namespace qd {

class CardDefinition
{
private:
  int64_t index;
  std::vector<CardEntryDefinition> card_entry_defs;

public:
  CardDefinition(int64_t _card_index, bool _optional);
  void add_card_entry_definition(CardEntryDefinition _entry);
};

} // namespace qd

#endif