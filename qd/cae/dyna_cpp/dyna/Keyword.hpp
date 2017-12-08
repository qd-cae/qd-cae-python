
#ifndef KEYWORD_HPP
#define KEYWORD_HPP

// includes
#include <cstdint>
#include <memory>
#include <string>

#include <dyna_cpp/dyna/Card.hpp>
#include <dyna_cpp/dyna/KeywordDefinition.hpp>

namespace qd {

class Keyword
{
private:
  std::shared_ptr<KeywordDefinition> definition;
  std::vector<std::shared_ptr<Card>> cards;

public:
  Keyword(std::shared_ptr<KeywordDefinition> _definition);
  void set_card_value(const std::string& _field_name, const std::string& value);
  void set_card_value(const std::string& _field_name, int64_t value);
  void set_card_value(const std::string& _field_name, double value);

  std::shared_ptr<Card> get_card(int64_t _index);
};

} // namespace qd

#endif