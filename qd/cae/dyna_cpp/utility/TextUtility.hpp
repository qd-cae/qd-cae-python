
#ifndef TEXTUTILITY_HPP
#define TEXTUTILITY_HPP

// includes
#include <algorithm>
#include <cctype>
#include <functional>
#include <locale>
#include <sstream>
#include <string>
#include <vector>

#include <iostream>

namespace qd {

/** Print a string as hex
 *
 * @param arg : string to be printed
 */
inline void
print_string_as_hex(const std::string& arg)
{
  for (auto c : arg)
    std::cout << std::hex << static_cast<int32_t>(c);
  std::cout << '\n';
}

/** Tests if a string has content (non spacing chars)
 *
 * @param _str : string to test
 * @return is_empty
 *
 * A string has no content, if it only contains
 * spaces, tabs etc.
 */
inline bool
str_has_content(const std::string& _str)
{
  return std::all_of(_str.begin(), _str.end(), [](const char c) {
    return std::isspace(static_cast<unsigned char>(c));
  });
}

/** Concat a vector of strings with line endings
 *
 * @param _lines
 * @return concat
 */
inline std::string
str_concat_lines(const std::vector<std::string>& _lines)
{
  std::stringstream ss;
  for (const auto& line : _lines)
    ss << line << '\n';

  // remove last linebreak
  auto ret = ss.str();
  ret.pop_back();

  return ret;
}

/** Trim a string from left
 *
 * @param s : string getting trimmed
 * @return trimmed_s : string trimmed
 */
inline std::string&
trim_left(std::string& s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
  return s;
}

/** Trim a string from right and copy it
 *
 * @param s : string getting trimmed
 * @return trimmed_s : string trimmed
 */
inline std::string
trim_right_copy(std::string s)
{
  s.erase(std::find_if(s.rbegin(),
                       s.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
            .base(),
          s.end());
  return std::move(s);
}

/** Trim a string from right
 *
 * @param s : string getting trimmed
 * @return trimmed_s : string trimmed
 */
inline std::string&
trim_right(std::string& s)
{
  s.erase(std::find_if(s.rbegin(),
                       s.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
            .base(),
          s.end());
  return s;
}

/** Trim a string from both sides and copy it
 *
 * @param s : string getting trimmed
 * @return trimmed_s : string trimmed
 */
inline std::string&
trim(std::string& s)
{
  trim_right(s);
  return trim_left(s);
}

/** Trim a string from both sides
 *
 * @param s : string getting trimmed
 * @return trimmed_s : string trimmed
 */
inline std::string
trim_copy(std::string s)
{
  trim_right(s);
  return trim_left(s);
}

/** Convert a string into a vector of lines
 *
 * @param _buffer string which has the lines
 * @param _ignore_trailing_lines : ignores trailing empty lines in the front
 * @return lines
 */
inline std::vector<std::string>
string_to_lines(const std::string& _buffer, bool _ignore_trailing_lines = false)
{
  std::string line;
  std::vector<std::string> lines;
  std::stringstream ss(_buffer, std::ios_base::in);

  if (_ignore_trailing_lines) {
    while (std::getline(ss, line) &&
           std::all_of(line.begin(), line.end(), [](char c) {
             return std::isspace(static_cast<unsigned char>(c));
           }))
      continue;
    if (!line.empty())
      lines.push_back(line);
  }

  while (std::getline(ss, line))
    lines.push_back(line);
  return lines;
}

/** Convert a string to lower-case (by reference)
 *
 * @param _txt
 */
inline std::string&
to_lower(std::string& _txt)
{
  std::transform(_txt.begin(), _txt.end(), _txt.begin(), ::tolower);
  return _txt;
}

/** Convert a string to lower-case (creates copy)
 *
 * @param _txt
 */
inline std::string
to_lower_copy(std::string _txt)
{
  std::transform(_txt.begin(), _txt.end(), _txt.begin(), ::tolower);
  return std::move(_txt);
}

/** Convert string into some type
 * @param T value : value to convert to string
 * @return string result
 */
template<typename T>
T
string_to_type(std::string const& str)
{
  T value;
  std::stringstream ss(str);
  ss >> value;
  return value;
}

/** Enum for classifying the data in strings
 *
 */
enum class StringType
{
  STRING,
  FLOAT,
  INTEGER
};

/** Get the type of a string
 *
 * @param _arg some string to test
 * @return type
 */
StringType
get_string_type(const std::string& _arg);

/** Extract numbers from a string
 * @param std::string _text
 */
template<typename T>
std::vector<T>
extract_integers(const std::string& text)
{

  std::vector<T> numbers;

  std::stringstream ss;
  ss << text;
  T found;
  std::string temp;

  while (getline(ss, temp, ' ')) {
    if (std::stringstream(temp) >> found) {
      numbers.push_back(found);
    }
  }

  return numbers;
}

/** Check if a string has only numbers
 * @param std::string _text : strig to check
 * @param size_t _pos = 0 : starting position
 * @return bool is_number
 *
 * Also returns true if the string is empty.
 */
bool
string_has_only_numbers(const std::string& _text, size_t start_pos);

/** Convert a char buffer to a buffer of lines
 *
 * @param _data : char buffer
 * @return lines : line buffer
 */
inline std::vector<std::string>
convert_chars_to_lines(const std::vector<char>& _data)
{

  std::stringstream st(std::string(_data.begin(), _data.end()),
                       std::ios_base::in);

  std::string line;
  std::vector<std::string> lines;
  while (std::getline(st, line))
    lines.push_back(line);

  return lines;
}

/** Check if a string ends with another string
 *
 * @param value value to check the ending of
 * @param ending ending to check
 * @return is_ending
 */
inline bool
ends_with(std::string const& value, std::string const& ending)
{
  if (ending.size() > value.size())
    return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

/** Get a word from a string
 *
 * @param _begin begin iterator
 * @param _end ending iterator
 *
 * Uses regex.
 */
std::string
get_word(std::string::const_iterator _begin, std::string::const_iterator _end);

/** Get a word from a string
 *
 * @param _begin begin iterator
 * @param _end ending iterator
 *
 * Uses regex.
 */
std::ptrdiff_t
get_word_position(const std::string& _str,
                  const std::string& _pattern = "\\w+");

} // namespace qd

#endif
