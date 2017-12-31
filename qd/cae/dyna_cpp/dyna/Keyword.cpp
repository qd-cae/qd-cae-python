
#include <iostream>
#include <stdexcept>

#include <dyna_cpp/dyna/Keyword.hpp>
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
 * @param _line_number line index in the file for ordering
 *
 * This constructor is meant for fast creation while parsing.
 * The data is redundant, since the keyword name should also be
 * in the lines argument.
 */
Keyword::Keyword(const std::vector<std::string>& _lines, int64_t _line_number)
  : line_number(_line_number)
  , lines(_lines)
{

  // field size
  if (ends_with(get_keyword_name(), "+")) {
    field_size = 20;
  } else {
    field_size = 10;
  }
}

/** Construct a keyword
 *
 * @param _lines lines of the buffer
 * @param _keyword_name name of the keyword (redundant!)
 * @param _line_number line index in the file for ordering
 *
 * This constructor is meant for fast creation while parsing.
 * The data is redundant, since the keyword name should also be
 * in the lines argument.
 */
Keyword::Keyword(const std::vector<std::string>& _lines,
                 const std::string& _keyword_name,
                 int64_t _line_number)
  : line_number(_line_number)
  , lines(_lines)
{
  // field size
  if (ends_with(_keyword_name, "+")) {
    field_size = 20;
  } else {
    field_size = 10;
  }
}

/** Switches the field size between single and double size
 *
 * Single size are 10 characters, Long is 20 characters.
 * Beware: Also the first comment line above fields will
 *         be translated. This should be the field names
 *
 */
void
Keyword::switch_field_size()
{

  // new sizes
  auto old_field_size = field_size;
  field_size = old_field_size == 10 ? 20 : 10;

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
      change_line_field_size(iLine, old_field_size, field_size);
      if (is_comment(lines[iLine - 1]))
        change_line_field_size(iLine - 1, old_field_size, field_size);
    }
  } // for iLine
}

/** Change the field size of the specified line
 *
 * @param iLine line to change the field size
 * @param old_field_size old size of the fields
 * @param new_field_size new size of the fields ... obviously
 */
void
Keyword::change_line_field_size(size_t _iLine,
                                size_t old_field_size,
                                size_t new_field_size)
{

  // checks
  if (old_field_size == new_field_size)
    return;

  auto& line = lines[_iLine];
  auto _is_comment = is_comment(line);

  // COPY: FIELDS
  if (!_is_comment) {

    // allocate new line
    size_t new_size = static_cast<size_t>(static_cast<double>(new_field_size) /
                                          static_cast<double>(old_field_size) *
                                          static_cast<double>(line.size()));

    auto new_line = std::string(new_size, ' ');

    auto copy_size =
      new_field_size > old_field_size ? old_field_size : new_field_size;

    // copy full fields
    size_t iField;
    for (iField = 0; iField < line.size() / old_field_size; ++iField) {
      auto start = line.begin() + iField * old_field_size;
      auto end = start + copy_size;
      std::copy(start, end, new_line.begin() + iField * new_field_size);
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
    for (iField = 0; iField < old_line.size() / old_field_size; ++iField) {

      // search word pattern
      auto start = old_line.begin() + iField * old_field_size;
      auto end = start + old_field_size;
      auto comment_name = get_word(start, end);

      // Pattern found (take first only)
      if (!comment_name.empty())
        set_comment_name_unchecked(_iLine, iField, comment_name);
    }

    // Reallocate line
    if (line.size() > new_field_size * iField)
      line.resize(new_field_size * iField);
  }
}

/** Set a new line buffer
 *
 * @param _new_lines
 *
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

  throw(std::invalid_argument("Can not find field:" + _keyword_name +
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
    if (line[0] == '*')
      return line;
  }
  throw(
    std::runtime_error("Can not find keyword name in the keyword buffer (must "
                       "begin with * in first column)."));
}

/** Set a card value from its card and field index
 *
 * @param iCard card index (non comment lines)
 * @param iField field index (a field has 10/20 chars)
 * @param _value value to set
 *
 * The values not fitting into the field will be cut off.
 * Index checks will not be performed!
 */
void
Keyword::set_card_value_unchecked(size_t _iLine,
                                  size_t _iField,
                                  const std::string& _value,
                                  size_t _field_size)
{
  auto& line = lines[_iLine];

  // check for user-specific field size
  _field_size = _field_size != 0 ? _field_size : field_size;

  // clear field or allocate
  clear_field(line, _iField, _field_size);

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
  line.replace(start, len, _value, 0, len);
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
Keyword::set_comment_name_unchecked(size_t _iLine,
                                    size_t _iField,
                                    const std::string& _name,
                                    size_t _field_size)
{
  auto& line = lines[_iLine];
  if (!is_comment(line))
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
      c_start = _field_size * _iField + delimiter_size; // first is separator

      break;
    case (Keyword::Align::MIDDLE):
      c_start = _field_size * _iField + _field_size / 2 - name_len / 2;
      break;
    default: // RIGHT
      c_start = _field_size * (_iField + 1) - name_len;
      break;
  }

  // empty field
  clear_field(line, _iField, _field_size);

  // assign value to field
  line.replace(c_start, name_len, _name, 0, name_len);
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