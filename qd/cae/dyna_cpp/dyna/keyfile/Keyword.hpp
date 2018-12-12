
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

  enum class KeywordType
  {
    GENERIC,
    NODE,
    ELEMENT,
    PART,
    INCLUDE_PATH,
    INCLUDE
  };

  // Static settings
  static bool name_delimiter_used;
  static char name_delimiter;
  static char name_spacer;
  static Align name_alignment;
  static Align field_alignment;

protected:
  Keyword::KeywordType kw_type;
  size_t field_size;              // size of the fields (8 or 20)
  int64_t position;               // line index in file (keeps order)
  std::vector<std::string> lines; // line buffer

  // as always, dirty stuff is better kept private ...
  inline bool is_comment(const std::string& _line) const;
  inline bool is_keyword(const std::string& _line) const;
  template<typename T>
  inline size_t iChar_to_iField(T _iChar) const;
  template<typename T>
  size_t iCard_to_iLine(T _iCard, bool _auto_extend = true);
  template<typename T>
  std::string get_field_byLine(const std::string& _line,
                               T _iField,
                               size_t _field_size = 0) const;

  std::pair<size_t, size_t> get_field_indexes(
    const std::string& _field_name) const;

  void set_card_value_byLine(std::string& _line,
                             size_t _iField,
                             const std::string& _value,
                             size_t _field_size = 0);
  void set_card_name_byLine(std::string& _line,
                            size_t _iField,
                            const std::string& _name,
                            size_t _field_size = 0);
  inline std::string get_card_value_byLine(const std::string& _line,
                                           size_t _iField,
                                           size_t _field_size = 0);
  void change_field_size_byLine(size_t iLine,
                                size_t old_field_size,
                                size_t new_field_size);
  template<typename T>
  void reformat_field_byLine(std::string& _line,
                             T _iField,
                             size_t _field_size = 0);
  void reformat_line(std::string& _line, size_t _field_size = 0);
  void clear_field(std::string& _line, size_t iField, size_t _field_size = 0);

public:
  explicit Keyword(const std::string& _lines,
                   int64_t _position = 0,
                   size_t _field_size = 0);
  explicit Keyword(const std::vector<std::string>& _lines,
                   int64_t _position = 0,
                   size_t _field_size = 0);
  explicit Keyword(const std::vector<std::string>& _lines,
                   const std::string& _keyword_name,
                   int64_t _position = 0,
                   size_t _field_size = 0);

  // getters
  inline size_t get_field_size() const;
  template<typename T>
  inline void set_field_size(T _new_field_size);

  inline KeywordType get_keyword_type() const;
  static KeywordType determine_keyword_type(const std::string& str);
  std::string get_keyword_name() const;

  inline bool has_long_fields() const;
  inline size_t size();
  inline int64_t get_position() const;
  /*
  bool contains_field(const std::string& _name) const;
  */
  virtual std::string str();
  void print();

  // card stuff
  template<typename T>
  void switch_field_size(const std::vector<T> _skip_cards);
  template<typename T>
  void reformat_all(std::vector<T> _skip_cards);
  template<typename T>
  void reformat_card_value(T _iCard,
                           T _iField,
                           size_t _field_size = 0,
                           bool _format_field = true,
                           bool _format_name = true);
  template<typename T>
  inline std::string get_card(T _iCard);
  inline std::string get_card_value(const std::string& _field_name,
                                    size_t _field_size = 0);
  template<typename T>
  inline std::string get_card_value(T _iCard,
                                    T _iField,
                                    size_t _field_size = 0);
  template<typename T>
  inline void set_card(T _iCard, const std::string& _data);
  inline void set_card_value(const std::string& _field_name,
                             const std::string& _value,
                             size_t _field_size = 0);
  inline void set_card_value(const std::string& _field_name,
                             int64_t _value,
                             size_t _field_size = 0);
  inline void set_card_value(const std::string& _field_name,
                             double _value,
                             size_t _field_size = 0);
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

  // lines
  void append_line(const std::string& _new_line);
  void set_lines(const std::vector<std::string>& _new_lines);
  template<typename T>
  void set_line(T iLine, const std::string& _line);
  template<typename T>
  void insert_line(T iLine, const std::string& _line);
  template<typename T>
  void remove_line(T iLine);
  inline std::vector<std::string> get_lines() const;
  inline std::vector<std::string>& get_lines();
  template<typename T>
  inline const std::string& get_line(T _iLine) const;

  inline void set_position(int64_t _iLine);

  // useful but mostly internal stuff
  inline size_t get_line_index_of_next_card(size_t _iLineOffset = 0);
};

/** Get the current field size in chars
 *
 */
size_t
Keyword::get_field_size() const
{
  return field_size;
}

/** Set a new field size
 *
 */
template<typename T>
void
Keyword::set_field_size(T _new_field_size)
{
  check_non_negative(_new_field_size);
  field_size = static_cast<size_t>(_new_field_size);
}

/** Get the type of the keyword
 *
 * @return type
 *
 * Generic, Node, etc.
 */
inline Keyword::KeywordType
Keyword::get_keyword_type() const
{
  return kw_type;
}

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
Keyword::get_card_value_byLine(const std::string& _line,
                               size_t _iField,
                               size_t _field_size)
{
  return trim_copy(get_field_byLine(_line, _iField, _field_size));
}

/** Get an entire card from a card index
 *
 * @oaram _iCard : card index
 * @return line : card as string
 */
template<typename T>
std::string
Keyword::get_card(T _iCard)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  return lines[iCard_to_iLine(_iCard, false)];
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
  return get_card_value_byLine(lines[iLine], _iField, _field_size);
}

/** Get a card value by its name in the comments
 *
 * @param _field_name name of the field
 * @return field, is empty if field does not exist
 */
std::string
Keyword::get_card_value(const std::string& _field_name, size_t _field_size)
{
  auto indexes = get_field_indexes(_field_name);
  return get_card_value_byLine(
    lines[indexes.first], indexes.second, _field_size);
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
Keyword::get_position() const
{
  return position;
}

/** Set the line number at which this block shall be positioned in the file
 *
 * @param _iLine line number
 *
 * If two blocks are dangered to overlap, they will be positioned
 * in sequence. Thus thus line number is rather seen as a wish.
 */
void
Keyword::set_position(int64_t _iLine)
{
  position = _iLine;
}

/** Checks if a string is a comment
 *
 * @param _line line to check
 * @return boolean whether the line is a comment
 */
bool
Keyword::is_comment(const std::string& _line) const
{
  return _line.empty() ? false : _line[0] == '$';
}

/** Query whether the fields are twice as long
 * @return
 */
bool
Keyword::has_long_fields() const
{
  return get_keyword_name().find('+') != std::string::npos;
}

/** Checks if a string is a keyword
 *
 * @param _line line to check
 * @return boolean whether the line is a keyword
 */
bool
Keyword::is_keyword(const std::string& _line) const
{
  return _line.empty() ? false : _line[0] == '*';
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
    // if (!lines[index].empty() && lines[index][0] != '$' &&
    //     lines[index][0] != '*')
    if (!is_comment(lines[index]) && !is_keyword(lines[index])) {
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

/** find the line index of the next card
 *
 * @param _iCard : card index
 * @return index : returns lines.size() if none found
 *
 * Does not check the line at the offset position.
 */
size_t
Keyword::get_line_index_of_next_card(size_t _iLineOffset)
{

  for (size_t iLine = _iLineOffset + 1; iLine < lines.size(); ++iLine) {
#ifdef QD_DEBUG
    if (iLine >= lines.size())
      throw(std::invalid_argument("iLine > lines.size()"));
#endif
    if (!is_comment(lines[iLine]) && !is_keyword(lines[iLine]))
      return iLine;
  }
  return lines.size();
}

/** Get a field index from a char index
 *
 * @param char_index
 * @return field index
 *
 * TODO: make field_size an argument
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
  return std::move(_line.substr(start, _field_size));
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
Keyword::get_lines()
{
  return lines;
}

/** Switches the field size between single and double size
 *
 * Single size are 10 characters, Long is 20 characters.
 * Beware: Also the first comment line above fields will
 *         be translated. This should be the field names
 *
 */
template<typename T>
void
Keyword::switch_field_size(const std::vector<T> _skip_cards)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");
  static_assert(std::is_unsigned<T>::value, "Unsigned number required.");

  // new sizes
  auto old_field_size = field_size;
  field_size = old_field_size <= 10 ? old_field_size * 2 : old_field_size / 2;

  T iCard = 0;
  for (size_t iLine = 0; iLine < lines.size(); ++iLine) {
    auto& line = lines[iLine];

    // enlarge: add + at the end
    if (is_keyword(line) && field_size > old_field_size) {
      if (line[line.size() - 1] == ' ')
        line[line.size() - 1] = '+';
      else
        line += '+';
      continue;
    }
    // shorten: remove all +
    else {
      std::replace(line.begin(), line.end(), '+', ' ');
    }

    // is it a card?
    if (!is_comment(line) && !is_keyword(line)) {

      if (std::find(_skip_cards.begin(), _skip_cards.end(), iCard++) !=
          _skip_cards.end()) {
        continue;
      }

      change_field_size_byLine(iLine, old_field_size, field_size);
      if (is_comment(lines[iLine - 1]))
        change_field_size_byLine(iLine - 1, old_field_size, field_size);
    }
  } // for iLine
}

/** Set a card/line
 *
 * @param _iCard : index of card
 * @param _data : string data to set
 */
template<typename T>
void
Keyword::set_card(T _iCard, const std::string& _data)
{
  set_line(iCard_to_iLine(_iCard), _data);
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
  set_card_value_byLine(
    lines[indexes.first], indexes.second, _value, _field_size);
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
    if (iLine != 0 && !is_comment(lines[iLine - 1])) {
      insert_line(iLine++, "$");
    }

    // do the thing
    set_card_name_byLine(
      lines[iLine - 1], iField_u, _comment_name, _field_size_);
  }

  // set value
  set_card_value_byLine(lines[iLine], iField_u, _value, _field_size_);
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

/** Reformat the whole card according to the formatting rules
 *
 * @param _skip_cards indexes of cards to skip
 *
 * // TODO: skip_cards argument
 */
template<typename T>
void
Keyword::reformat_all(std::vector<T> _skip_cards)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");
  static_assert(std::is_unsigned<T>::value, "Unsigned number required.");

  T iCard = 0;
  for (size_t iLine = 0; iLine < lines.size(); ++iLine) {
    auto& line = lines[iLine];

    // is it a card?
    if (!is_comment(line) && !is_keyword(line)) {

      if (std::find(_skip_cards.begin(), _skip_cards.end(), iCard++) !=
          _skip_cards.end())
        continue;

      reformat_line(line);

      // has the card a comment header?
      if (iLine > 0 && is_comment(lines[iLine - 1]))
        reformat_line(lines[iLine - 1]);
    }
  } // FOR:iLine
}

/** Reformat a specific field of a line
 *
 * @param _line line to format
 * @param _iField field index
 * @param _field_size size of the field
 */
template<typename T>
void
Keyword::reformat_field_byLine(std::string& _line,
                               T _iField,
                               size_t _field_size)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");
  static_assert(std::is_unsigned<T>::value, "Unsigned number required.");

  // COMMENTS
  if (is_comment(_line)) {
    auto val = get_word(_line.begin() + _iField * _field_size,
                        _line.begin() + (_iField + 1) * _field_size);
    // if (!val.empty())
    set_card_name_byLine(_line, _iField, val, _field_size);
  }
  // KEYWORD
  else {
    auto val = get_card_value_byLine(_line, _iField, _field_size);
    // if (!val.empty())
    set_card_value_byLine(_line, _iField, val, _field_size);
  }
}

/** Reformat a field of a specific line
 *
 * @param _iLine index of line to format
 * @param _iField index of field to reformat
 * @param _field_size size of the field to reformat
 */
template<typename T>
void
Keyword::reformat_card_value(T _iCard,
                             T _iField,
                             size_t _field_size,
                             bool format_field,
                             bool format_name)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  auto iLine = iCard_to_iLine(_iCard, true);

  // card
  if (format_field)
    reformat_field_byLine(lines[iLine], _iField, _field_size);

  // name
  if (format_name && iLine > 0 && is_comment(lines[iLine - 1]))
    reformat_field_byLine(lines[iLine - 1], _iField, _field_size);
}

} // namespace qd

#endif
