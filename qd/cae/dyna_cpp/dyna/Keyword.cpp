
#include <stdexcept>

#include <dyna_cpp/dyna/Keyword.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

namespace qd {

/** Construct a keyword from a definition
 *
 * @param _definition definition of the Keyword.
 */
Keyword::Keyword(const std::string& _keyword_name,
                 int64_t _line_number,
                 const std::vector<std::string>& _lines)
  : line_number(_line_number)
  , lines(_lines)
  , keyword_name(_keyword_name)
{
  // field size
  if (ends_with(_keyword_name, "+"))
    field_size = 20;
  else
    field_size = 10;
}

/** Checks if a string is a comment
 *
 * @param _line line to check
 * @return boolean whether the line is a comment
 */
inline bool
Keyword::is_comment(const std::string& _line) const
{
  return _line[0] == '$';
}

/** Get the buffer of a keyword field
 *
 * @param _keyword_name name of the keyword to search from comments
 * @returns index buffer index, is negative if it can not be found
 */
std::pair < Keyword::get_field_indexes(const std::string& _keyword_name) const
{

  for (size_t index = 0; index < lines.size() - 1; ++index) {
    const auto& line = lines[index];
    const auto& next_line = lines[index + 1];

    // only comments can contain the field names
    // continues if next line is also a comment, since
    // the field may not be comment obviously
    if (is_comment(line) && is_comment(next_line))
      continue;

    // field name in comment line
    size_t start = line.find(_keyword_name);
    if (start != std::string::npos)
      return std::make_pair(index + 1, get_field_col(start));
  }
}

throw(std::invalid_argument("Can not find field:" + _keyword_name +
                            " in comments."));
}

/** Get a field index from a char index
 *
 * @param char_index
 * @return field index
 */
inline int64_t
Keyword::get_field_col(size_t _char_index) const
{
  return _char_index / field_size;
}

/** Get the index of a card entry
 *
 * @param iCard index of the card
 * @return index index in the lines buffer
 */
int64_t
Keyword::get_card_index(size_t iCard) const
{

  // search index
  size_t nCards = 0;
  for (size_t index = 0; index < lines.size(); ++index) {
    if (lines[index][0] != '$') {
      ++nCards;
      if (nCards == iCard)
        return index;
    }
  }

  // simply append more empty lines
  lines.resize(lines.size() + iCard - nCards);

  return static_cast<int64_t>(lines.size());
}

/** Get the name of the keyword
 *
 * @return name name of the keyword
 */
std::string
Keyword::get_keyword_name() const
{
  return this->keyword_name;
}

/** Get a copy of the line buffer of the keyword
 *
 * @return lines copy of line buffer
 */
inline std::vector<std::string>
Keyword::get_lines() const
{
  return lines;
}

/** Get the line buffer of the keyword
 *
 * @return lines line buffer
 */
inline std::vector<std::string>&
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
inline void
Keyword::set_card_value(const std::string& _field_name,
                        const std::string& _value)
{
  auto indexes = get_field_index(_field_name);

  set_card_value(indexes.first, indexes.second, _value);
}

/** Set a card value from it's name
 *
 * @param _field_name name of the variable field
 * @param value value to set
 *
 * The field name will be searched in the comment lines.
 */
inline void
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
inline void
Keyword::set_card_value(const std::string& _field_name, double _value)
{
  set_card_value(_field_name, std::to_string(_value));
}

/** Set a card value from its card and field index
 *
 * @param iCard card index (non comment lines)
 * @param iField field index (a field has 8/20 chars)
 * @param _value value to set
 *
 * The values not fitting into the field will be cut off.
 * Index checks will not be performed!
 */
void
Keyword::set_card_value_unchecked(int64_t line_index,
                                  int64_t char_index,
                                  const std::string& _value)
{
  auto& line = lines[line_index];

  // insert spaces if original string was too small
  size_t tmp = static_cast<size_t>(char_index + field_size);
  if (line.size() < tmp) {
    line.insert(line.size(), tmp - line.size(), " ");
  }

  // assign value to field
  if (_value.size() > field_size)
    line.replace(char_index, field_size, _value);
  else
    line.replace(char_index, _value.size(), _value);
}

/** Set a card value from its card and field index
 *
 * @param iCard card index (non comment lines)
 * @param iField field index (a field has 10/20 chars)
 * @param _value value to set
 *
 * The values not fitting into the field will be cut off.
 */
void
Keyword::set_card_value(int64_t iCard,
                        int64_t iField,
                        const std::string& _value)
{
  auto line_index = get_card_index(iCard);
  set_card_value_unchecked(line_index, iField * field_size, _value);
}

/** Set a card value from its card and field index
 *
 * @param iCard card index (non comment lines)
 * @param iField field index (a field has 10/20 chars)
 * @param _value value to set
 *
 * The values not fitting into the field will be cut off.
 */
void
Keyword::set_card_value(int64_t iCard, int64_t iField, int64_t _value)
{
  auto line_index = get_card_index(iCard);
  set_card_value_unchecked(
    line_index, iField * field_size, std::to_string(_value));
}

/** Set a card value from its card and field index
 *
 * @param iCard card index (non comment lines)
 * @param iField field index (a field has 10/20 chars)
 * @param _value value to set
 *
 * The values not fitting into the field will be cut off.
 */
void
Keyword::set_card_value(int64_t iCard, int64_t iField, double _value)
{
  auto line_index = get_card_index(iCard);
  set_card_value_unchecked(
    line_index, iField * field_size, std::to_string(_value));
}

} // namespace qd