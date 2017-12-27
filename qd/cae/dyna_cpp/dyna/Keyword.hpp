
#ifndef KEYWORD_HPP
#define KEYWORD_HPP

// includes
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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
  std::pair<int64_t, int64_t> get_field_indexes(
    const std::string& _field_name) const;
  int64_t get_card_index(size_t iCard, bool auto_extend = false);
  int64_t iChar_to_iField(size_t char_index) const;
  void set_card_value_unchecked(int64_t line_index,
                                int64_t char_index,
                                const std::string& _value);

public:
  Keyword(const std::string& _keyword_name,
          int64_t _line_number,
          const std::vector<std::string>& _lines);

  // getters
  std::string get_keyword_name() const;
  inline std::vector<std::string> get_lines() const;
  inline std::vector<std::string>& get_line_buffer();

  // setters
  // template<typename T>
  // void set_card_value(const std::string& _field_name, T _value);

  void set_card_value(const std::string& _field_name,
                      const std::string& _value);
  void set_card_value(const std::string& _field_name, int64_t _value);
  void set_card_value(const std::string& _field_name, double _value);
  void set_card_value(int64_t iCard,
                      int64_t iField,
                      const std::string& _value,
                      const std::string& _comment_name = "");
  void set_card_value(int64_t iCard,
                      int64_t iField,
                      int64_t _value,
                      const std::string& _comment_name = "");
  void set_card_value(int64_t iCard,
                      int64_t iField,
                      double _value,
                      const std::string& _comment_name = "");
  void insert(size_t iLine, const std::string& _line);
  void remove(size_t iLine);
  std::string str();
  void print();
};

/*
template<typename T>
void
Keyword::set_card_value(const std::string& _field_name, T _value)
{
  static_assert(0, "Can not set card value for the specified type.");
}

template<>
void
Keyword::set_card_value<>(const std::string& _field_name, T _value)
{
  static_assert(std::is_integral(T)::value, "Wrong template function used.");

}
*/

/** Get a copy of the line buffer of the keyword
 *
 * @return lines copy of line buffer
 */
std::vector<std::string>
Keyword::get_lines() const
{
  return lines;
}

/** Get the line buffer of the keyword
 *
 * @return lines line buffer
 */
std::vector<std::string>&
Keyword::get_line_buffer()
{
  return lines;
}

} // namespace qd

#endif