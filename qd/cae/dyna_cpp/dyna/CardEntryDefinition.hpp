
#ifndef CARDENTRYDEFINITION_HPP
#define CARDENTRYDEFINITION_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>


namespace qd {

// Class
class CardEntryDefinition
{
public:
  // Variable Types
  enum class VariableType
  {
    NONE,
    INT32,
    INT64,
    FLOAT32,
    FLOAT64,
    STRING
  };

private:
  int64_t char_start;
  int64_t char_width;
  std::string card_entry_name;
  VariableType var_type;
  std::string default_value;
  bool is_convertible(const std::string& _string, VariableType _var_type);

public:
  CardEntryDefinition(int64_t _char_start,
                      int64_t _char_width,
                      VariableType _var_type,
                      const std::string& _card_entry_name = "",
                      const std::string& _default_value = "");
  void set_char_start(int64_t _start);
  void set_char_width(int64_t _width);
  void set_card_entry_name(const std::string _name);
  void set_variable_type(VariableType _type);
  void set_default_value(const std::string& value);

  int64_t get_char_start() const;
  int64_t get_char_width() const;
  std::string get_card_entry_name() const;
  VariableType get_variable_type() const;
  std::string get_default_value() const;

  void validate_uniqueness(std::shared_ptr<CardEntryDefinition> _def);
};

} // namespace qd

#endif