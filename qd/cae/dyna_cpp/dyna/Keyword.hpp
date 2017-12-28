
#ifndef KEYWORD_HPP
#define KEYWORD_HPP

// includes
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <dyna_cpp/utility/MathUtility.hpp>

namespace qd {

class Keyword
{
private:
  static char comment_delimiter;
  static char comment_spacer;

  int64_t field_size;             // size of the fields (8 or 20)
  int64_t line_number;            // line index in file (keeps order)
  std::vector<std::string> lines; // line buffer

  inline bool is_comment(const std::string& _line) const;
  inline bool is_keyword(const std::string& _line) const;
  // dirty
  std::pair<int64_t, int64_t> get_field_indexes(
    const std::string& _field_name) const;
  int64_t get_card_index(size_t iCard, bool auto_extend = false);
  int64_t iChar_to_iField(size_t char_index) const;
  void set_card_value_unchecked(int64_t line_index,
                                int64_t char_index,
                                const std::string& _value);
  void set_comment_name_unchecked(size_t iLine,
                                  size_t iField,
                                  const std::string& _name);
  void change_line_field_size(size_t iLine,
                              size_t old_field_size,
                              size_t new_field_size);
  void clear_field(std::string& _line, size_t iField);

public:
  // Keyword(const std::vector<std::string>& _lines, int64_t _line_number = 0);
  Keyword(const std::vector<std::string>& _lines,
          const std::string& _keyword_name,
          int64_t _line_number = 0);

  // getters
  std::string get_keyword_name() const;
  inline std::vector<std::string> get_lines() const;
  inline std::vector<std::string>& get_line_buffer();
  std::string str();
  void print();

  // setters
  void switch_field_size();
  inline void set_card_value(const std::string& _field_name,
                             const std::string& _value);
  inline void set_card_value(const std::string& _field_name, int64_t _value);
  inline void set_card_value(const std::string& _field_name, double _value);

  // card values by indexes
  inline void set_card_value(int64_t iCard,
                             int64_t iField,
                             const std::string& _value,
                             const std::string& _comment_name = "");
  inline void set_card_value(int64_t iCard,
                             int64_t iField,
                             int64_t _value,
                             const std::string& _comment_name = "");
  inline void set_card_value(int64_t iCard,
                             int64_t iField,
                             double _value,
                             const std::string& _comment_name = "");

  template<typename T>
  void set_line(T iLine, const std::string& _line);
  template<typename T>
  void insert_line(T iLine, const std::string& _line);
  template<typename T>
  void remove_line(T iLine);
  void set_comment_delimiter(char new_delimiter);
  void set_comment_spacer(char new_delimiter);
};

/** Checks if a string is a comment
 *
 * @param _line line to check
 * @return boolean whether the line is a comment
 */
bool
Keyword::is_comment(const std::string& _line) const
{
  return _line[0] == '$';
}

/** Checks if a string is a keyword
 *
 * @param _line line to check
 * @return boolean whether the line is a keyword
 */
bool
Keyword::is_keyword(const std::string& _line) const
{
  return _line[0] == '*';
}

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

/** Set a card value from it's name
 *
 * @param _field_name name of the variable field
 * @param value value to set
 *
 * The field name will be searched in the comment lines.
 */
void
Keyword::set_card_value(const std::string& _field_name,
                        const std::string& _value)
{
  auto indexes = get_field_indexes(_field_name);
  set_card_value_unchecked(indexes.first, indexes.second * field_size, _value);
}

/** Set a card value from it's name
 *
 * @param _field_name name of the variable field
 * @param value value to set
 *
 * The field name will be searched in the comment lines.
 */
void
Keyword::set_card_value(const std::string& _field_name, int64_t _value)
{
  set_card_value(_field_name, std::to_string(_value));
}

/** Set a card value from it's name
 *
 * @param _field_name name of the variable field
 * @param value value to set
 *
 * The field name will be searched in the comment lines.
 */
void
Keyword::set_card_value(const std::string& _field_name, double _value)
{
  std::stringstream ss;
  ss.precision(field_size - 1);
  ss << _value;
  set_card_value(_field_name, ss.str());
}

/** Set a card value from its card and field index
 *
 * @param iCard card index (non comment lines)
 * @param iField field index (a field has 10/20 chars)
 * @param _value value to set
 * @param _comment_name name to set in the comments
 *
 * The values not fitting into the field will be cut off.
 */
void
Keyword::set_card_value(int64_t iCard,
                        int64_t iField,
                        const std::string& _value,
                        const std::string& _comment_name)
{
  auto line_index = get_card_index(iCard);

  // comment treatment
  if (!_comment_name.empty()) {

    // create a comment line if neccessary
    if (line_index == 0 || !is_comment(lines[line_index - 1])) {
      insert_line(line_index, "$");
      ++line_index;
    }

    // do the thing
    set_comment_name_unchecked(line_index - 1, iField, _comment_name);
  }

  set_card_value_unchecked(line_index, iField * field_size, _value);
}

/** Set a card value from its card and field index
 *
 * @param iCard card index (non comment lines)
 * @param iField field index (a field has 10/20 chars)
 * @param _value value to set
 * @param _comment_name name to set in the comments
 *
 * The values not fitting into the field will be cut off.
 */
void
Keyword::set_card_value(int64_t iCard,
                        int64_t iField,
                        int64_t _value,
                        const std::string& _comment_name)
{
  set_card_value(iCard, iField, std::to_string(_value), _comment_name);
}

/** Set a card value from its card and field index
 *
 * @param iCard card index (non comment lines)
 * @param iField field index (a field has 10/20 chars)
 * @param _value value to set
 * @param _comment_name name to set in the comments
 *
 * The values not fitting into the field will be cut off.
 */
void
Keyword::set_card_value(int64_t iCard,
                        int64_t iField,
                        double _value,
                        const std::string& _comment_name)
{
  std::stringstream ss;
  ss.precision(field_size - 1);
  ss << _value;
  set_card_value(iCard, iField, ss.str(), _comment_name);
}

/** Set the text of a specific line
 *
 * @param iLine line index where to replace
 * @param _line replacement string
 */
template<typename T>
void
Keyword::set_line(T iLine, const std::string& _line)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  if (static_cast<size_t>(iLine) > lines.size())
    lines.resize(iLine + 1);

  lines[iLine] = _line;
}

/** Insert a line into the buffer
 *
 * @param iLine line index where to insert
 * @param _line line to insert
 */
template<typename T>
void
Keyword::insert_line(T iLine, const std::string& _line)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  if (static_cast<size_t>(iLine) > lines.size()) {
    lines.resize(iLine + 1);
    lines[iLine] = _line;
  } else {
    lines.insert(lines.begin() + iLine, _line);
  }
}

/** Remove a line in the buffer
 *
 * @param iLine line to remove
 */
template<typename T>
void
Keyword::remove_line(T iLine)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  if (static_cast<size_t>(iLine) > lines.size())
    return;

  lines.erase(lines.begin() + iLine);
}

} // namespace qd

#endif