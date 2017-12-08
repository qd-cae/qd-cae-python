
#ifndef KEYWORDDEFINITION_HPP
#define KEYWORDDEFINITION_HPP

// includes
#include <memory>
#include <string>
#include <vector>

#include <dyna_cpp/dyna/CardDefinition.hpp>

namespace qd {

class KeywordDefinition
{
private:
  std::string name;
  std::vector<CardDefinition> card_defs;

public:
  KeywordDefinition(const std::string& _name);
  void add_card_definition(CardDefinition _card_def);
};

} // namespace qd

#endif