
#ifndef KEYWORD_HPP
#define KEYWORD_HPP

// includes
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qd {

class Keyword
{
private:
  int64_t field_size;
  std::string keyword_name;
  int64_t line_number;
  std::vector<std::string> lines;

  bool is_comment(const std::string& _line) const;
  std::tuple<int64_t, int64_t> get_field_indexes(
    const std::string& _field_name) const;
  int64_t get_field_col(size_t char_index) const;
  int64_t get_card_index(size_t iCard) const;
  void set_card_value_unchecked(int64_t line_index,
                                int64_t char_index,
                                const std::string& _value);

public:
  Keyword(const std::string& _keyword_name,
          int64_t _line_number,
          const std::vector<std::string>& _lines);
  std::string get_keyword_name() const;
  std::vector<std::string> get_lines() const;
  std::vector<std::string>& get_line_buffer();

  void set_card_value(const std::string& _field_name, const std::string& value);
  void set_card_value(const std::string& _field_name, int64_t value);
  void set_card_value(const std::string& _field_name, double value);
  void set_card_value(int64_t iCard, int64_t iField, const std::string& _value);
  void set_card_value(int64_t iCard, int64_t iField, int64_t _value);
  void set_card_value(int64_t iCard, int64_t iField, double _value);
};

} // namespace qd

#endif