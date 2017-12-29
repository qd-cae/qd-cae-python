
#include <iostream>
#include <regex>
#include <stdexcept>

#include <dyna_cpp/dyna/Keyword.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

namespace qd {

// static vars
char Keyword::comment_delimiter = '|';
char Keyword::comment_spacer = '-';

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
Keyword::change_line_field_size(size_t iLine,
                                size_t old_field_size,
                                size_t new_field_size)
{

  // checks
  if (old_field_size == new_field_size)
    return;

  auto& line = lines[iLine];
  auto _is_comment = is_comment(line);

  // COPY: FIELDS
  if (!_is_comment) {

    // allocate new line
    size_t new_size = static_cast<size_t>(static_cast<double>(new_field_size) /
                                          static_cast<double>(old_field_size) *
                                          static_cast<double>(line.size()));

    auto new_line =
      _is_comment ? std::string(new_size, '-') : std::string(new_size, ' ');

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
    lines[iLine] = new_line;
  }
  // COPY: COMMENTS
  else {

    std::string old_line = line;

    std::regex pattern("\\w+");
    std::regex_iterator<std::string::iterator> it_end;

    size_t iField;
    for (iField = 0; iField < old_line.size() / old_field_size; ++iField) {

      // search word pattern
      auto start = old_line.begin() + iField * old_field_size;
      auto end = start + old_field_size;
      std::regex_iterator<std::string::iterator> it(start, end, pattern);

      // Pattern found (take first only)
      if (it != it_end)
        set_comment_name_unchecked(iLine, iField, it->str());
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
std::pair<int64_t, int64_t>
Keyword::get_field_indexes(const std::string& _keyword_name) const
{

  for (size_t index = 0; index < lines.size() - 1; ++index) {
    const auto& line = lines[index];
    const auto& next_line = lines[index + 1];

    // only comments can contain the field names
    // continues if next line is also a comment, since
    // the field may not be comment obviously
    if (!is_comment(line) || is_comment(next_line))
      continue;

    // field name in comment line
    size_t start = line.find(_keyword_name);
    if (start != std::string::npos)
      return std::make_pair(static_cast<int64_t>(index + 1),
                            iChar_to_iField(start));
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
Keyword::set_card_value_unchecked(int64_t line_index,
                                  int64_t char_index,
                                  const std::string& _value)
{
  auto& line = lines[line_index];

  // insert spaces if original string was too small
  size_t tmp = static_cast<size_t>(char_index + field_size);
  if (line.size() < tmp)
    line.resize(tmp, ' ');

  // assign value to field
  if (_value.size() >= static_cast<size_t>(field_size))
    line.replace(char_index, field_size, _value, 0, field_size);
  else
    line.replace(char_index, _value.size(), _value);
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
Keyword::set_comment_name_unchecked(size_t iLine,
                                    size_t iField,
                                    const std::string& _name)
{
  auto& line = lines[iLine];
  if (!is_comment(line))
    throw(std::invalid_argument("The specified line is not a comment line"));

  // compute offsets
  constexpr size_t delimiter_size = 1;

  size_t field_size_tmp = static_cast<size_t>(field_size);
  size_t name_len = _name.size() > field_size_tmp - delimiter_size
                      ? field_size_tmp - 1
                      : _name.size();
  size_t c_start = field_size_tmp * iField + field_size_tmp / 2 - name_len / 2;

  // check string size
  if (line.size() < c_start + field_size_tmp)
    line.resize(c_start + field_size_tmp, ' ');

  // empty field
  clear_field(line, iField);

  // assign value to field
  if (_name.size() >= static_cast<size_t>(field_size_tmp))
    line.replace(c_start, name_len, _name, 0, name_len);
  else
    line.replace(c_start, name_len, _name);

  // delimiter
  if (iField != 0)
    line[field_size_tmp * iField] = Keyword::comment_delimiter;
}

/** Clear a field
 *
 * @param iLine line to be cleared
 * @param iField field index to be cleared
 */
void
Keyword::clear_field(std::string& _line, size_t iField)
{
  auto is_comment_tmp = is_comment(_line);

  char spacer = is_comment_tmp ? Keyword::comment_spacer : ' ';

  auto field_size_tmp = static_cast<size_t>(field_size);
  auto start = iField * field_size_tmp;
  auto end = start + field_size_tmp;

  if (is_comment_tmp && iField == 0)
    _line[start] = '$';
  else if (is_comment_tmp)
    _line[start] = Keyword::comment_delimiter;
  else
    _line[start] = ' ';

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