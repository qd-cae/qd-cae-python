
#include <regex>

#include "TextUtility.hpp"

namespace qd {

/** Check if a string has only numbers
 * @param std::string _text : strig to check
 * @param size_t _pos = 0 : starting position
 * @return bool is_number
 *
 * Also returns true if the string is empty.
 */
bool
string_has_only_numbers(const std::string& _text, size_t start_pos)
{

  // Check
  if (_text.size() == start_pos) {
    return true;
  } else if (start_pos > _text.size()) {
    throw(
      std::invalid_argument("Starting pos is greater than the string length."));
  }
  if (_text.empty())
    return true;

  // iterate
  for (size_t ii = start_pos; ii < _text.size(); ++ii) {

    // return if one char was not a number
    if (!std::isdigit(_text[ii]))
      return false;
  }

  // Return true in case loop went through
  return true;
}

StringType
get_string_type(const std::string& _arg)
{
  size_t nPoints = 0;
  bool has_digit = false;
  bool is_exponential = false;

  for (size_t ii = 0; ii < _arg.size(); ++ii) {

    if (std::ispunct(_arg[ii])) {
      ++nPoints;
      if (nPoints > 1 || is_exponential)
        return StringType::STRING;
    }

    if (std::isdigit(_arg[ii]))
      has_digit = true;

    if (_arg[ii] == 'e' || _arg[ii] == 'E') {

      if (is_exponential)
        return StringType::STRING;
      is_exponential = true;

      if (ii + 1 < _arg.size()) {
        if (_arg[ii + 1] == '+' || _arg[ii + 1] == '-')
          ++ii;
        continue;
      } else {
        return StringType::STRING;
      }
    }

    if (!std::isdigit(_arg[ii]) && !std::isblank(_arg[ii]) &&
        !std::ispunct(_arg[ii]))
      return StringType::STRING;
  }

  if (has_digit) {
    if (nPoints != 0 || is_exponential)
      return StringType::FLOAT;
    else
      return StringType::INTEGER;
  } else
    return StringType::STRING;
}

std::string
get_word(std::string::const_iterator _begin, std::string::const_iterator _end)
{

  const std::regex pattern("\\w+");
  const std::sregex_iterator it_end;
  std::sregex_iterator it(_begin, _end, pattern);

  if (it != it_end)
    return it->str();

  return std::string();
}

std::ptrdiff_t
get_word_position(const std::string& _str, const std::string& _pattern)
{

  const std::regex pattern("\\b" + _pattern + "\\b");

  std::sregex_iterator it(_str.begin(), _str.end(), pattern), it_end;
  if (it != it_end) {
    return it->position();
  }

  return -1;
}

} // namespace qd