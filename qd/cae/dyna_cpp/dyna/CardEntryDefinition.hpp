
#ifndef CARDENTRYDEFINITION_HPP
#define CARDENTRYDEFINITION_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace qd {

// Variable Types
enum class VariableType
{
  INT32,
  INT64,
  FLOAT32,
  FLOAT64
};

// Class
class CardEntryDefinition
{
private:
  int64_t index;
  int64_t char_width;
  std::string name;
  VariableType var_type;
  std::vector<char> default_value;

public:
  CardEntryDefinition(int64_t _index,
                      int64_t width,
                      VariableType _var_type,
                      const std::string& _name = "");
  void set_default_value(char* _data, size_t _size_to_copy);

  int64_t get_index() const;
  int64_t get_char_width() const;
  std::string get_name() const;
  VariableType get_variable_type() const;
  std::vector<char> get_default_value() const;
};

} // namespace qd

#endif