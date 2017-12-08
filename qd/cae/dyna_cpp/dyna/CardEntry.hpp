
#ifndef CARDENTRY_HPP
#define CARDENTRY_HPP

// includes
#include <dyna_cpp/dyna/CardEntryDefinition.hpp>

#include <cstdint>
#include <memory>

namespace qd {

class CardEntry
{
private:
  std::string value;
  std::shared_ptr<CardEntryDefinition> definition;

public:
  CardEntry(std::shared_ptr<CardEntryDefinition> _definition);
  void set_value(std::string _value);

  std::string get_value() const;
  int64_t get_value_int() const;
  double get_value_dbl() const;
  std::string str() const;
};

} // namespace qd

#endif // CARDENTRY_HPP