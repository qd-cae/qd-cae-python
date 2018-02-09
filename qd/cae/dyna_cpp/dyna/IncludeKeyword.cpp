
#include <dyna_cpp/dyna/IncludeKeyword.hpp>
#include <dyna_cpp/dyna/KeyFile.hpp>

namespace qd {

/** Create an include keyword
 *
 * @param _parent_kf : parent keyfile (if loading mesh)
 * @param _lines : line buffer
 * @param _iLine : line index
 */
IncludeKeyword::IncludeKeyword(KeyFile* _parent_kf,
                               const std::vector<std::string> _lines,
                               int64_t _iLine)
  : Keyword(_lines, _iLine)
{}

/** Load the includes in the
 *
 *
 */
void
IncludeKeyword::load(bool _load_mesh)
{
  auto iLine = iCard_to_iLine(0, false);
  auto header_size = iLine;

  // create keywords
  for (; iLine < lines.size(); ++iLine) {
    const auto& line = lines[iLine];

    if (line.empty() || is_comment(line))
      break;

    auto fpath = parent_kf->resolve_include_filepath(line);
    auto kf = std::make_shared<KeyFile>(
      fpath,
      parent_kf->get_read_generic_keywords(),
      parent_kf->get_parse_mesh(),
      parent_kf->get_load_includes(),
      parent_kf->get_encryption_detection_threshold());

    kf->load(_load_mesh);
    includes.push_back(kf);
  }

  // handle trailing lines
  for (; iLine < lines.size(); ++iLine)
    trailing_lines.push_back(lines[iLine]);

  lines.resize(header_size);
}

/** Get the keyword as a string
 *
 * @return keyword as string
 */
std::string
IncludeKeyword::str()
{
  // build header
  std::stringstream ss;
  for (const auto& entry : lines)
    ss << entry << '\n';

  // data
  for (auto& include : includes)
    ss << include->get_filepath();

  // trailing lines
  for (const auto& line : trailing_lines)
    ss << line << '\n';

  return ss.str();
}

} // namespace:qd