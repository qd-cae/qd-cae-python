
#ifndef KEYWORD_HPP
#define KEYWORD_HPP

// includes
#include <algorithm>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <dyna_cpp/utility/MathUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

namespace qd {

class Keyword
{

public:
  enum class Align
  {
    LEFT,
    MIDDLE,
    RIGHT
  };

  // Static settings
  static bool name_delimiter_used;
  static char name_delimiter;
  static char name_spacer;
  static Align name_alignment;
  static Align field_alignment;

private:
  size_t field_size;              // size of the fields (8 or 20)
  int64_t line_index;             // line index in file (keeps order)
  std::vector<std::string> lines; // line buffer

  // as always, dirty stuff is better kept private ...
  inline bool is_comment(const std::string& _line) const;
  inline bool is_keyword(const std::string& _line) const;
  template<typename T>
  size_t iCard_to_iLine(T _iCard, bool _auto_extend = true);
  template<typename T>
  inline size_t iChar_to_iField(T _iChar) const;
  template<typename T>
  std::string get_field_byLine(const std::string& _line,
                               T _iField,
                               size_t _field_size = 0) const;

  std::pair<size_t, size_t> get_field_indexes(
    const std::string& _field_name) const;

  void set_card_value_byLine(size_t _iLine,
                             size_t _iField,
                             const std::string& _value,
                             size_t _field_size = 0);
  void set_card_name_byLine(size_t _iLine,
                            size_t _iField,
                            const std::string& _name,
                            size_t _field_size = 0);
  inline std::string get_card_value_byLine(size_t _iLine,
                                           size_t _iField,
                                           size_t _field_size = 0);
  void change_field_size_byLine(size_t iLine,
                                size_t old_field_size,
                                size_t new_field_size);
  void clear_field(std::string& _line, size_t iField, size_t _field_size = 0);

public:
  explicit Keyword(const std::vector<std::string>& _lines,
                   int64_t _line_index = 0);
  explicit Keyword(const std::string& _lines, int64_t _line_index = 0);
  explicit Keyword(const std::vector<std::string>& _lines,
                   const std::string& _keyword_name,
                   int64_t _line_index = 0);

  // getters
  std::string get_keyword_name() const;
  inline std::vector<std::string> get_lines() const;
  inline std::vector<std::string>& get_line_buffer();
  template<typename T>
  inline const std::string& get_line(T _iLine) const;
  inline std::string get_card_value(const std::string& _field_name);
  template<typename T>
  inline std::string get_card_value(T _iCard,
                                    T _iField,
                                    size_t _field_size = 0);
  inline bool has_long_fields() const;
  inline size_t size();
  inline int64_t get_line_index();
  /*
  bool contains_field(const std::string& _name) const;
  */
  std::string str();
  void print();

  // setters
  void switch_field_size(const std::vector<size_t> _cards_to_skip);
  inline void set_card_value(const std::string& _field_name,
                             const std::string& _value,
                             size_t _field_size = 0);
  inline void set_card_value(const std::string& _field_name,
                             int64_t _value,
                             size_t _field_size = 0);
  inline void set_card_value(const std::string& _field_name,
                             double _value,
                             size_t _field_size = 0);

  // card values by indexes
  template<typename T>
  inline void set_card_value(T iCard,
                             T iField,
                             const std::string& _value,
                             const std::string& _comment_name = "",
                             size_t _field_size = 0);
  template<typename T>
  inline void set_card_value(T iCard,
                             T iField,
                             int64_t _value,
                             const std::string& _comment_name = "",
                             size_t _field_size = 0);
  template<typename T>
  inline void set_card_value(T iCard,
                             T iField,
                             double _value,
                             const std::string& _comment_name = "",
                             size_t _field_size = 0);
  void set_lines(const std::vector<std::string>& _new_lines);
  template<typename T>
  void set_line(T iLine, const std::string& _line);
  template<typename T>
  void insert_line(T iLine, const std::string& _line);
  template<typename T>
  void remove_line(T iLine);
  inline void set_line_index(int64_t _iLine);
};

/** Get a specific line in the keyword
 *
 * @param _iLine  index of the line
 *
 * the index may also be negative (python style)
 */
template<typename T>
const std::string&
Keyword::get_line(T _iLine) const
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

/** Get a card value from a specific line and field
 *
 * @param _iLine line to get a field from
 * @param _iField field to get data from
 * @param _field_size optional field size
 * @return data string
 */
std::string
Keyword::get_card_value_byLine(size_t _iLine,
                               size_t _iField,
                               size_t _field_size)
{
  return trim(get_field_byLine(lines[_iLine], _iField, _field_size));
}

/** Get a card value from an index pair
 *
 * @param _iCard card index
 * @param _iField field index
 * @param _field_size size of the field
 * @return field, is empty if field does not exist
 */
template<typename T>
std::string
Keyword::get_card_value(T _iCard, T _iField, size_t _field_size)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  auto iLine = iCard_to_iLine(_iCard, false);
  return get_card_value_byLine(iLine, _iField, _field_size);
}

/** Get a card value by its name in the comments
 *
 * @param _field_name name of the field
 * @return field, is empty if field does not exist
 */
std::string
Keyword::get_card_value(const std::string& _field_name)
{
  auto indexes = get_field_indexes(_field_name);
  return get_card_value_byLine(indexes.first, indexes.second);
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
 * @return line_index line index of the keyword in the file
 */
int64_t
Keyword::get_line_index()
{
  return line_index;
}

/** Set the line number at which this block shall be positioned in the file
 *
 * @param _iLine line number
 *
 * If two blocks are dangered to overlap, they will be positioned
 * in sequence. Thus thus line number is rather seen as a wish.
 */
void
Keyword::set_line_index(int64_t _iLine)
{
  line_index = _iLine;
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
    if (lines[index].size() > 0 && lines[index][0] != '$' &&
        lines[index][0] != '*') {
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

  throw(std::invalid_argument("card index " + std::to_string(iCard_u) +
                              " >= number of cards " +
                              std::to_string(++nCards)));
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

/** Get a specific field from a line
 *
 * @param _line line to get the field from
 * @param _iField field index
 * @return field data of the field, empty if field exeeds line bounds
 */
template<typename T>
std::string
Keyword::get_field_byLine(const std::string& _line,
                          T _iField,
                          size_t _field_size) const
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  // field handling
  _iField = index_treatment(_iField, iChar_to_iField(_line.size()));
  auto iField_u = static_cast<size_t>(_iField);

  // use individual field_size?
  _field_size = _field_size != 0 ? _field_size : field_size;

  // do the thing
  auto start = iField_u * _field_size < _line.size() ? iField_u * _field_size
                                                     : _line.size();

  // pewpew results
  return _line.substr(start, _field_size);
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
                        const std::string& _value,
                        size_t _field_size)
{
  auto indexes = get_field_indexes(_field_name);
  set_card_value_byLine(indexes.first, indexes.second, _value, _field_size);
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
                        int64_t _value,
                        size_t _field_size_)
{
  set_card_value(_field_name, std::to_string(_value), _field_size_);
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
                        double _value,
                        size_t _field_size_)
{
  std::stringstream ss;
  ss.precision(field_size - 1);
  ss << _value;
  set_card_value(_field_name, ss.str(), _field_size_);
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
template<typename T>
void
Keyword::set_card_value(T _iCard,
                        T _iField,
                        const std::string& _value,
                        const std::string& _comment_name,
                        size_t _field_size_)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  if (_iField < 0)
    throw(std::invalid_argument("field index may not be negative!"));

  auto iLine = iCard_to_iLine(_iCard, true);
  auto iField_u = static_cast<size_t>(_iField);

  // set comment name
  if (!_comment_name.empty()) {

    // create a comment line if neccessary
    if (iLine == 0 || !is_comment(lines[iLine - 1])) {
      insert_line(iLine++, "$");
    }

    // do the thing
    set_card_name_byLine(iLine - 1, iField_u, _comment_name, _field_size_);
  }

  // set value
  set_card_value_byLine(iLine, iField_u, _value, _field_size_);
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
template<typename T>
void
Keyword::set_card_value(T iCard,
                        T iField,
                        int64_t _value,
                        const std::string& _comment_name,
                        size_t _field_size)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  set_card_value(
    iCard, iField, std::to_string(_value), _comment_name, _field_size);
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
template<typename T>
void
Keyword::set_card_value(T iCard,
                        T iField,
                        double _value,
                        const std::string& _comment_name,
                        size_t _field_size)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");
  _field_size = _field_size != 0 ? _field_size : field_size;

  std::stringstream ss;
  ss.precision(_field_size - 1);
  ss << _value;
  set_card_value(iCard, iField, ss.str(), _comment_name, _field_size);
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
