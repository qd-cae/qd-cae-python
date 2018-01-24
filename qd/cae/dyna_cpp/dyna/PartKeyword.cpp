
#include <dyna_cpp/dyna/PartKeyword.hpp>

namespace qd {

/** Constructor of a part keyword
 *
 * @param _db_parts : parent database
 * @param _lines : keyword lines
 * @param _iLine : line index in file
 */
PartKeyword::PartKeyword(DB_Parts* _db_parts,
                         const std::vector<std::string>& _lines,
                         int64_t _iLine)
  : Keyword(_lines, _iLine)
  , db_parts(_db_parts)
{
  // TODO parse keyword
}

/** Get the keyword as a string
 *
 */
std::string
PartKeyword::str() const
{
  std::stringstream ss;
  for (const auto& entry : lines)
    ss << entry << '\n';

  ss.precision(7);
  // TODO: write parts

  for (const auto& entry : trailing_lines)
    ss << entry << '\n';

  return ss.str();
}

} // namespace:qd