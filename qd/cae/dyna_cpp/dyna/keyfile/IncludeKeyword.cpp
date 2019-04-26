
#include <dyna_cpp/dyna/keyfile/IncludeKeyword.hpp>
#include <dyna_cpp/dyna/keyfile/KeyFile.hpp>

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
  , parent_kf(_parent_kf)
{}

/** Load the includes internally
 *
 *
 */
void
IncludeKeyword::load()
{
  load(parent_kf->get_parse_mesh());
}

/** Load the include files
 *
 * @param _load_mesh : overwrite to prevent loading the mesh internally
 */
void
IncludeKeyword::load(bool _load_mesh)
{
  auto iLine = get_line_index_of_next_card(0);
  auto header_size = iLine;

  if (iLine == lines.size())
    return;

#ifdef QD_DEBUG
  std::cout << "Loading Include ...\n";
#endif

  // update parent search dirs
  auto master = parent_kf->get_master_keyfile();
  // auto dirs = master->get_include_dirs(true);
  auto dirs = parent_kf->get_include_dirs(true);

  // create keywords
  for (; iLine < lines.size(); ++iLine) {
    const auto& line = lines[iLine];

    if (line.empty() || is_comment(line))
      break;

    // auto fpath = master->resolve_include_filepath(line);
    auto fpath = parent_kf->resolve_include_filepath(line);
    auto kf = std::make_shared<KeyFile>(fpath,
                                        parent_kf->get_read_generic_keywords(),
                                        parent_kf->get_parse_mesh(),
                                        parent_kf->get_load_includes(),
                                        parent_kf);

    auto is_ok = kf->load(_load_mesh);
    if (is_ok) {
      includes.push_back(kf);
      unresolved_filepaths.push_back(line);
    }
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
  for (const auto& include_path : unresolved_filepaths)
    ss << include_path << '\n';

  // trailing lines
  for (const auto& line : trailing_lines)
    ss << line << '\n';

  return ss.str();
}

} // namespace:qd