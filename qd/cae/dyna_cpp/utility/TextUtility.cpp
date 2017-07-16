
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

  // Hardcoded stuff rulzZz
  const std::vector<char> number_chars = { '0', '1', '2', '3', '4',
                                           '5', '6', '7', '8', '9' };

  // iterate
  bool char_is_number = false;
  for (size_t ii = start_pos; ii < _text.size(); ++ii) {
    char_is_number = false;
    for (size_t jj = 0; jj < number_chars.size(); ++jj) {
      if (_text[ii] == number_chars[jj]) {
        char_is_number = true;
        break;
      }
    }

    // return if one char was not a number
    if (!char_is_number)
      return false;
  }

  // Return true in case loop went through
  return true;
}

} // namespace qd