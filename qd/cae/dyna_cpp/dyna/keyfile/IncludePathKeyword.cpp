
#include <dyna_cpp/dyna/keyfile/IncludePathKeyword.hpp>

namespace qd {

/** Constructor of an include dir keyword
 *
 * @param _lines
 * @param _iLine
 *
 * include dirs are dirs where an include may lie
 */
IncludePathKeyword::IncludePathKeyword(const std::vector<std::string> _lines,
                                       int64_t _iLine)
  : Keyword(_lines, _iLine)
{}

/** If the path keyword contains a relative path
 *
 * @return is_relative
 */
bool
IncludePathKeyword::is_relative() const
{
  auto name = get_keyword_name();
  to_lower(name);
  if (name.compare(0, 17, "*include_path_rel") == 0)
    return true;
  else
    return false;
}

/** Get the include dir paths
 *
 * @return dir_filepaths
 */
std::vector<std::string>
IncludePathKeyword::get_include_dirs()
{
  std::vector<std::string> dir_filepaths;
  for (auto iLine = get_line_index_of_next_card(0); iLine < lines.size();
       ++iLine) {
    auto line = trim_copy(lines[iLine]);

    // stop if an empty line is hit
    if (line.empty())
      break;

    dir_filepaths.push_back(line);
  }

  return dir_filepaths;
}

} // namespace:qd