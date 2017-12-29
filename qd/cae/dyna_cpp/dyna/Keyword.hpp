
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

  // as always, dirty stuff is better kept private ...
  inline bool is_comment(const std::string& _line) const;
  inline bool is_keyword(const std::string& _line) const;
  template<typename T>
  size_t iCard_to_iLine(T _iCard, bool _auto_extend = true);
  template<typename T>
  inline size_t iChar_to_iField(T _iChar) const;
  /*
  template<typename T>
  std::string get_iField(T iField) const;
  */

  std::pair<int64_t, int64_t> get_field_indexes(
    const std::string& _field_name) const;

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
  explicit Keyword(const std::vector<std::string>& _lines,
                   int64_t _line_number = 0);
  explicit Keyword(const std::vector<std::string>& _lines,
                   const std::string& _keyword_name,
                   int64_t _line_number = 0);

  // getters
  template<typename T>
  const std::string& operator[](T _iLine) const;
  /*
  template<typename T>
  const std::string& operator[](std::pair<T, T> _indexes) const;
  */
  inline const std::string& operator[](const std::string& _field_name);
  std::string get_keyword_name() const;
  inline std::vector<std::string> get_lines() const;
  inline std::vector<std::string>& get_line_buffer();
  inline std::string get_card_value(const std::string& _field_name);
  /*
  template<typename T>
  inline std::string get_card_value(std::pair<T, T> _indexes);
  */
  inline bool has_long_fields() const;
  inline size_t size();
  inline int64_t get_line_number();
  /*
  bool contains_field(const std::string& _name) const;
  */
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
  void set_lines(const std::vector<std::string>& _new_lines);
  template<typename T>
  void set_line(T iLine, const std::string& _line);
  template<typename T>
  void insert_line(T iLine, const std::string& _line);
  template<typename T>
  void remove_line(T iLine);
  inline void set_line_number(int64_t _iLine);

  // static functions
  static void set_comment_delimiter(char new_delimiter)
  {
    Keyword::comment_delimiter = new_delimiter;
  };
  static void set_comment_spacer(char new_spacer)
  {
    Keyword::comment_spacer = new_spacer;
  };
  static char get_comment_delimiter() { return Keyword::comment_delimiter; };
  static char get_comment_spacer() { return Keyword::comment_spacer; };
};

/** Get a specific line in the keyword
 *
 * @param _iLine  index of the line
 *
 * the index may also be negative (python style)
 */
template<typename T>
const std::string& Keyword::operator[](T _iLine) const
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  // negative index treatment
  _iLine = index_treatment(_iLine, lines.size());

  // test size
  if (_iLine > static_cast<T>(lines.size()))
    throw(std::invalid_argument(
      "line index:" + std::to_string(_iLine) +
      " exceeds number of lines:" + std::to_string(lines.size())));

  return lines[_iLine];
}

/** Get a card value from an index pair
 *
 * @param indexes, first is card index, second is field index
 */
/*
template<typename T>
std::string
get_card_value(std::pair<T, T> _indexes)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  auto iLine = iCard_to_iLine(_indexes.first);
  //auto iChar =
  // TODO
}
*/

/** Get a card value by its name in the comments
 *
 * @param _field_name name of the field
 */
std::string
Keyword::get_card_value(const std::string& _field_name)
{
  auto indexes = get_field_indexes(_field_name);
  return lines[indexes.first].substr(indexes.second * field_size, field_size);
}

/** Get the number of lines in the line buffer
 *
 * @return size number of lines in the line buffer
 */
size_t
Keyword::size()
{
  return lines.size();
}

/** Get the line number at which the block was in the text file
 *
 * @return line_number line index of the keyword in the file
 */
int64_t
Keyword::get_line_number()
{
  return line_number;
}

/** Set the line number at which this block shall be positioned in the file
 *
 * @param _iLine line number
 *
 * If two blocks are dangered to overlap, they will be positioned
 * in sequence. Thus thus line number is rather seen as a wish.
 */
void
Keyword::set_line_number(int64_t _iLine)
{
  line_number = _iLine;
}

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

/** Query whether the fields are twice as long
 * @return
 */
bool
Keyword::has_long_fields() const
{
  return field_size == 20;
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

/** Get the index of a card entry
 *
 * @param iCard index of the card
 * @return index index in the lines buffer
 */
template<typename T>
size_t
Keyword::iCard_to_iLine(T _iCard, bool _auto_extend)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  _iCard = index_treatment(_iCard, lines.size());
  auto iCard_u = static_cast<size_t>(_iCard);

  // search index
  size_t nCards = -1;
  for (size_t index = 0; index < lines.size(); ++index) {
    if (lines[index][0] != '$' && lines[index][0] != '*') {
      ++nCards;
      if (nCards == iCard_u)
        return index;
    }
  }

  // simply append more empty lines
  if (_auto_extend) {
    lines.resize(lines.size() + iCard_u - nCards);
    return lines.size() - 1;
  }

  throw(
    std::invalid_argument("card index:" + std::to_string(iCard_u) +
                          " >= number of cards:" + std::to_string(++nCards)));
}

/** Get a field index from a char index
 *
 * @param char_index
 * @return field index
 */
template<typename T>
size_t
Keyword::iChar_to_iField(T _iChar) const
{
  static_assert(std::is_integral<T>::value, "Integer number required.");
  static_assert(std::is_unsigned<T>::value, "Unsigned number required.");
  return static_cast<size_t>(_iChar) / field_size;
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
  auto line_index = iCard_to_iLine(iCard);

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