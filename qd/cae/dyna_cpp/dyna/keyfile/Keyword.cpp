
#include <iostream>
#include <stdexcept>

#include <dyna_cpp/dyna/keyfile/Keyword.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

namespace qd {

// static vars

bool Keyword::name_delimiter_used = false;
char Keyword::name_delimiter = '|';
char Keyword::name_spacer = ' ';
Keyword::Align Keyword::name_alignment = Keyword::Align::LEFT;
Keyword::Align Keyword::field_alignment = Keyword::Align::LEFT;

/** Construct a keyword
 *
 * @param _lines lines of the buffer
 * @param _position line index for keeping order
 *
 */
Keyword::Keyword(const std::vector<std::string>& _lines,
                 int64_t _position,
                 size_t _field_size)
  : kw_type(KeywordType::GENERIC)
  , position(_position)
  , lines(_lines)
{

  // field size
  if (_field_size == 0) {
    if (ends_with(get_keyword_name(), "+")) {
      field_size = 20;
    } else {
      field_size = 10;
    }
  } else {
    field_size = _field_size;
  }
}

/** Construct a keyword
 *
 * @param _lines the data of the keyword as a single string
 * @param _position line index for keeping order
 */
Keyword::Keyword(const std::string& _lines,
                 int64_t _position,
                 size_t _field_size)
  : Keyword(
      [](const std::string& _lines) {
        return string_to_lines(_lines, true);
      }(_lines),
      _position,
      _field_size)
{}

/** Construct a keyword
 *
 * @param _lines lines of the buffer
 * @param _keyword_name name of the keyword (redundant!)
 * @param _position line index in the file for ordering
 *
 * This constructor is meant for fast creation while parsing.
 * The data is redundant, since the keyword name should also be
 * in the lines argument.
 */
Keyword::Keyword(const std::vector<std::string>& _lines,
                 const std::string& _keyword_name,
                 int64_t _position,
                 size_t _field_size)
  : kw_type(KeywordType::GENERIC)
  , position(_position)
  , lines(_lines)
{
  // field size
  if (_field_size == 0) {
    if (ends_with(get_keyword_name(), "+")) {
      field_size = 20;
    } else {
      field_size = 10;
    }
  } else {
    field_size = _field_size;
  }
}

/** Get the type of the keyword
 *
 * @param str : keyword name as string
 * @return type
 *
 * Generic, Node, etc.
 */
Keyword::KeywordType
Keyword::determine_keyword_type(const std::string& _str)
{
  auto str_lower = to_lower_copy(_str);

  // *NODE
  if (str_lower.compare(0, 5, "*node") == 0) {
    if (str_lower.size() <= 5 || // just *node
        str_lower.compare(5, 13, "_scalar_value") == 0 ||
        str_lower.compare(5, 14, "_rigid_surface") == 0 ||
        str_lower.compare(5, 6, "_merge") == 0)
      return KeywordType::NODE;

  }
  // *ELEMENT
  else if (str_lower.compare(0, 8, "*element") == 0 && str_lower.size() > 8) {

    // *ELEMENT_BEAM
    if (str_lower.compare(8, 5, "_beam") == 0)
      return KeywordType::ELEMENT;

    // *ELEMENT_SOLID
    if (str_lower.compare(8, 6, "_solid") == 0)
      return KeywordType::ELEMENT;

    // *ELEMENT_SHELL
    if (str_lower.compare(8, 6, "_shell") == 0) {

      if (str_lower.size() <= 14)
        return KeywordType::ELEMENT;

      if (str_lower.compare(14, 10, "_thickness") == 0 ||
          str_lower.compare(14, 5, "_beta") == 0 ||
          str_lower.compare(14, 5, "_mcid") == 0 ||
          str_lower.compare(14, 7, "_offset") == 0 ||
          str_lower.compare(14, 4, "_dof") == 0)
        return KeywordType::ELEMENT;
    }

    // *ELEMENT_TSHELL
    if (str_lower.compare(8, 7, "_tshell") == 0) {

      if (str_lower.size() <= 15)
        return KeywordType::ELEMENT;

      if (str_lower.compare(15, 5, "_beta") == 0)
        return KeywordType::ELEMENT;
    }
  }
  // *PART
  else if (str_lower.compare(0, 5, "*part") == 0) {
    if (str_lower.size() > 5) {

      // unsupported part options
      if (str_lower.compare(5, 9, "_adaptive") == 0 ||
          str_lower.compare(5, 7, "_anneal") == 0 ||
          str_lower.compare(5, 10, "_composite") == 0 ||
          str_lower.compare(5, 10, "_duplicate") == 0 ||
          str_lower.compare(5, 6, "_modes") == 0 ||
          str_lower.compare(6, 5, "_move") == 0 ||
          str_lower.compare(5, 7, "_sensor") == 0 ||
          str_lower.compare(5, 8, "_stacked") == 0)
        return KeywordType::GENERIC;
      else
        return KeywordType::PART;
    } else {
      return KeywordType::PART;
    }
  }
  // *INCLUDE
  else if (str_lower.compare(0, 8, "*include") == 0) {

    // TODO
    if (str_lower.size() == 8 || str_lower.compare(8, 4, "_binary") == 0)
      return KeywordType::INCLUDE;

    // PATH
    if (str_lower.compare(8, 5, "_path") == 0) {

      if (str_lower.size() == 13 || str_lower.compare(13, 9, "_relative") == 0)
        return KeywordType::INCLUDE_PATH;
    }
  }
  // *INCLUDE_PATH
  else if (str_lower.compare(0, 13, "*include_path") == 0) {
  }

  return KeywordType::GENERIC;
}

/** Change the field size of the specified line
 *
 * @param iLine line to change the field size
 * @param old_field_size old size of the fields
 * @param new_field_size new size of the fields ... obviously
 */
void
Keyword::change_field_size_byLine(size_t _iLine,
                                  size_t old_field_size,
                                  size_t new_field_size)
{

  // checks
  if (old_field_size == new_field_size)
    return;

  auto& line = lines[_iLine];

  // COPY: FIELDS
  if (!is_comment(line)) {

    // allocate new line
    size_t new_size = static_cast<size_t>(static_cast<double>(new_field_size) /
                                          static_cast<double>(old_field_size) *
                                          static_cast<double>(line.size()));

    auto new_line = std::string(new_size, ' ');

    // copy full fields
    size_t iField;
    size_t nFields = line.size() / old_field_size;
    for (iField = 0; iField < nFields; ++iField) {
      auto val = get_card_value_byLine(line, iField, old_field_size);
      set_card_value_byLine(new_line, iField, val, new_field_size);
    }
    // copy rest
    auto rest = line.size() % old_field_size;
    if (rest != 0)
      std::copy(line.end() - rest,
                line.end(),
                new_line.begin() + iField * new_field_size);

    // assign
    lines[_iLine] = new_line;
  }
  // COPY: COMMENTS
  else {

    std::string old_line = line;

    size_t iField;
    size_t nFields = old_line.size() / old_field_size;
    size_t rest = old_line.size() % old_field_size;
    for (iField = 0; iField < nFields; ++iField) {

      // search word pattern
      auto start = old_line.begin() + iField * old_field_size;
      auto end = start + old_field_size;
      auto comment_name = get_word(start, end);

      // Pattern found (take first only)
      if (!comment_name.empty())
        set_card_name_byLine(line, iField, comment_name, new_field_size);
    }

    // is there any rest?
    if (rest != 0) {
      auto comment_name =
        get_word(old_line.begin() + iField * old_field_size, old_line.end());
      if (!comment_name.empty())
        set_card_name_byLine(line, iField++, comment_name, new_field_size);
    }

    // Reallocate line
    if (line.size() > new_field_size * iField)
      line.resize(new_field_size * iField);
  }
}

/** Reformat a specific line
 *
 * @param _line line to reformat
 *
 * The formatting uses the global formatting settings
 */
void
Keyword::reformat_line(std::string& _line, size_t _field_size)
{

  // auto line_is_comment = is_comment(_line);
  _field_size = _field_size != 0 ? _field_size : field_size;
  auto nFields = _line.size() / field_size;
  auto rest = _line.size() % _field_size;

  for (size_t iField = 0; iField < nFields; ++iField) {
    reformat_field_byLine(_line, iField, _field_size);
  }

  if (rest != 0) {
    reformat_field_byLine(_line, nFields, _field_size);
  }
}

/** Append a new line to the string buffer
 *
 * @param _new_line
 */
void
Keyword::append_line(const std::string& _new_line)
{
  lines.push_back(_new_line);
}

/** Set a new line buffer
 *
 * @param _new_lines
 */
void
Keyword::set_lines(const std::vector<std::string>& _new_lines)
{

  // check if a keyword can be found
  for (const auto& line : _new_lines)
    if (is_keyword(line)) {
      lines = _new_lines;
      return;
    }

  throw(std::invalid_argument(
    "Can not find a keyword beginning with * in new line buffer."));
}

/** Get the buffer of a keyword field
 *
 * @param _keyword_name name of the keyword to search from comments
 * @returns indexes first index is is the line, second is the field
 */
std::pair<size_t, size_t>
Keyword::get_field_indexes(const std::string& _keyword_name) const
{

  for (size_t iLine = 0; iLine < lines.size() - 1; ++iLine) {
    const auto& line = lines[iLine];
    const auto& next_line = lines[iLine + 1];

    // only comments can contain the field names
    // continues if next line is also a comment, since
    // the field may not be comment obviously
    if (!is_comment(line) || is_comment(next_line))
      continue;

    // field name in comment line
    auto start = get_word_position(line, _keyword_name);
    if (start >= 0)
      return std::make_pair(iLine + 1,
                            iChar_to_iField(static_cast<size_t>(start)));
  }

  throw(std::invalid_argument("Can not find field: " + _keyword_name +
                              " in comments."));
}

/** Get the name of the keyword
 *
 * @return name name of the keyword
 */
std::string
Keyword::get_keyword_name() const
{
  for (const auto& line : lines) {
    if (is_keyword(line)) {
      if (line.back() == '+')
        return line.substr(0, line.size() - 1);
      else
        return line;
    }
  }
  return std::string();
}

/** Set a card value from its card and field index
 *
 * @param iCard card index (non comment lines)
 * @param iField field index
 * @param _value value to set
 *
 * The values not fitting into the field will be cut off.
 * Index checks will not be performed!
 */
void
Keyword::set_card_value_byLine(std::string& _line,
                               size_t _iField,
                               const std::string& _value,
                               size_t _field_size)
{

  // check for user-specific field size
  _field_size = _field_size != 0 ? _field_size : field_size;

  // clear field or allocate
  clear_field(_line, _iField, _field_size);

  // copy range
  auto len = std::min(_field_size, _value.size());

  // where to put it?
  size_t start;
  switch (field_alignment) {
    case (Keyword::Align::LEFT):
      start = _iField * _field_size;
      break;
    case (Keyword::Align::MIDDLE):
      start = _field_size * _iField + _field_size / 2 - len / 2;
      break;
    default: // Right
      start = (_iField + 1) * _field_size - len;
      break;
  }

  // assign value to field
  _line.replace(start, len, _value, 0, len);
}

/** Set the name of a field in the comments
 *
 * @param iLine line in the buffer
 * @param iField field index to set
 * @param _name name to set
 *
 * Does not check for wrong line index.
 * If the name is too large, it will be cropped.
 * Throws if line is not a comment.
 */
void
Keyword::set_card_name_byLine(std::string& _line,
                              size_t _iField,
                              const std::string& _name,
                              size_t _field_size)
{
  if (!is_comment(_line))
    throw(std::invalid_argument("The specified line is not a comment line"));

  // check for user-specific field size
  _field_size = _field_size != 0 ? _field_size : field_size;

  // First field always has an delimiter offset due to comment symbol
  const size_t delimiter_size = Keyword::name_delimiter_used || _iField == 0;

  // compute offsets
  size_t name_len = _name.size() > _field_size - delimiter_size
                      ? _field_size - delimiter_size
                      : _name.size();

  size_t c_start;
  switch (Keyword::name_alignment) {
    case (Keyword::Align::LEFT):
      c_start = _field_size * _iField + delimiter_size;
      break;
    case (Keyword::Align::MIDDLE):
      c_start = _field_size * _iField + _field_size / 2 - name_len / 2;
      break;
    default: // RIGHT
      c_start = _field_size * (_iField + 1) - name_len;
      break;
  }

  // empty field
  clear_field(_line, _iField, _field_size);

  // assign value to field
  _line.replace(c_start, name_len, _name, 0, name_len);
}

/** Clear a field
 *
 * @param _line line to be cleared
 * @param _iField field index to be cleared
 */
void
Keyword::clear_field(std::string& _line, size_t _iField, size_t _field_size)
{
  auto is_comment_tmp = is_comment(_line);

  // check for user-specific field size
  _field_size = _field_size != 0 ? _field_size : field_size;

  // choose spacer
  char spacer = is_comment_tmp ? Keyword::name_spacer : ' ';

  // Hmm what could this be?
  auto start = _iField * _field_size;
  auto end = start + _field_size;

  if (is_comment_tmp && _iField == 0)
    _line[start] = '$';
  else if (is_comment_tmp && Keyword::name_delimiter_used)
    _line[start] = Keyword::name_delimiter;
  else
    _line[start] = ' ';

  // check line size
  if (end > _line.size())
    _line.resize(end);

  // Killer loop
  for (size_t iChar = start + 1; iChar < end; ++iChar)
    _line[iChar] = spacer;
}

/** Get the keyword as a string
 *
 * @return keyword as string
 */
std::string
Keyword::str()
{
  std::stringstream ss;
  for (const auto& entry : lines)
    ss << entry << '\n';
  return ss.str();
}

/** Print the card
 *
 */
void
Keyword::print()
{
  for (const auto& entry : lines)
    std::cout << entry << '\n';
  std::cout << std::flush;
}

} // namespace qd