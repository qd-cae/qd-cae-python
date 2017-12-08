
#ifndef CARD_HPP
#define CARD_HPP

// includes
#include <cstdint>
#include <string>
#include <vector>

#include <dyna_cpp/dyna/CardDefinition.hpp>
#include <dyna_cpp/dyna/CardEntry.hpp>

namespace qd {

class Card
{
private:
  int64_t line_number;
  std::shared_ptr<CardDefinition> definition;
  std::vector<std::shared_ptr<CardEntry>> card_entries;

public:
  Card(std::shared_ptr<CardDefinition> definition);
  void set_line_number(int64_t _line_number);
  void set_card_value(const std::string& _field_name, const std::string& value);
  void set_card_value(const std::string& _field_name, int64_t value);
  void set_card_value(const std::string& _field_name, double value);

  std::string str(bool print_info = true) const;
  int64_t get_line_number() const;
  std::shared_ptr<CardEntry> get_card_entry(const std::string& _name);
  template<typename T>
  T get_card_value(const std::string& _name) const;
};

} // namepspace qd

#endif CARD_HPP