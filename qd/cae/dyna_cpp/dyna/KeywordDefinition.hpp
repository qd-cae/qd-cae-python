
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
  std::string keyword_name;
  std::vector<std::shared_ptr<CardDefinition>> card_defs;

public:
  KeywordDefinition(const std::string& _name);
  void add_card_definition(std::shared_ptr<CardDefinition> _card_def);

  int64_t get_nCards() const;
  std::shared_ptr<CardDefinition> get_card_definition(int64_t _iCard);
  std::string get_keyword_name() const;
};

} // namespace qd

#endif