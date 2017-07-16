
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

namespace qd {

/** Convert some type into a string.
 * @param T value : value to convert to string
 * @return string result
 */
/*
template<typename T>
std::string
to_string(T const& value)
{
  std::ostringstream os;
  os << value;
  return os.str();
}
*/

/** Trim a string from left and copy it
 *
 * @param s : string getting trimmed
 * @return trimmed_s : string trimmed
 */
inline std::string
trim_left_copy(std::string s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
  return s;
}

/** Trim a string from left
 *
 * @param s : string getting trimmed
 * @return trimmed_s : string trimmed
 */
inline std::string
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
  return s;
}

/** Trim a string from right
 *
 * @param s : string getting trimmed
 * @return trimmed_s : string trimmed
 */
inline std::string
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
inline std::string
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

/** Preprocess a string for dyna.
 * @param string s : string to preprocess
 * @return string& s : string with removed comments
 */
inline std::string
preprocess_string_dyna(std::string _text)
{

  size_t pos = _text.find('$');
  if (pos != std::string::npos)
    return _text.substr(0, pos);
  return _text;
}

} // namespace qd

#endif
