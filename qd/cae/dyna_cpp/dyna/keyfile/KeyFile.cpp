
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/db/Part.hpp>
#include <dyna_cpp/dyna/keyfile/ElementKeyword.hpp>
#include <dyna_cpp/dyna/keyfile/IncludePathKeyword.hpp>
#include <dyna_cpp/dyna/keyfile/KeyFile.hpp>
#include <dyna_cpp/dyna/keyfile/NodeKeyword.hpp>
#include <dyna_cpp/dyna/keyfile/PartKeyword.hpp>
#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

namespace qd {

/**
 * Constructor for a LS-Dyna input file.
 */
KeyFile::KeyFile(bool _read_generic_keywords,
                 bool _parse_mesh,
                 bool _load_includes,
                 KeyFile* _parent_kf)
  : FEMFile("")
  , parent_kf(_parent_kf != nullptr ? _parent_kf : this)
  , load_includes(_load_includes)
  , read_generic_keywords(_read_generic_keywords)
  , parse_mesh(_parse_mesh)
  , has_linebreak_at_eof(true)
  , max_position(0)
{}

/** Constructor for reading a LS-Dyna input file.
 *
 * @param string filepath : filepath of a key file to read
 */
KeyFile::KeyFile(const std::string& _filepath,
                 bool _read_generic_keywords,
                 bool _parse_mesh,
                 bool _load_includes,
                 KeyFile* _parent_kf)
  : FEMFile(_filepath)
  , parent_kf(_parent_kf != nullptr ? _parent_kf : this)
  , load_includes(_load_includes)
  , read_generic_keywords(_read_generic_keywords)
  , parse_mesh(_parse_mesh)
  , has_linebreak_at_eof(true)
  , max_position(0)
{}

/** Parse a keyfile
 *
 * @param _load_mesh : whether the mesh shall loaded
 * @return success : whether loading the data was successful
 *
 * The parameter can be used to prevent the loading of the mesh,
 * even though we use parse_mesh. We need this for includes.
 */
bool
KeyFile::load(bool _load_mesh)
{

  // read file
  auto my_filepath = resolve_include_filepath(get_filepath());
  std::vector<char> char_buffer = read_binary_file(my_filepath);
  has_linebreak_at_eof = char_buffer.back() == '\n';

#ifdef QD_DEBUG
  std::cout << "Specified filepath: " << get_filepath() << std::endl;
  std::cout << "Resolved  filepath: " << my_filepath << std::endl;
#endif

  // init parallel worker if master file
  // if (parent_kf == this)
  //   _wq.init_workers(1);

  // convert buffer into blocks
  size_t iLine = 0;
  std::string last_keyword;
  std::vector<std::string> line_buffer;
  std::vector<std::string> line_buffer_tmp;
  bool found_pgp_section = false;

  std::string line;
  auto string_buffer = std::string(char_buffer.begin(), char_buffer.end());
  std::stringstream st(string_buffer);
  // for (; std::getline(st, line); ++iLine) {
  for (; std::getline(st, line); ++iLine) {

    if (line.find("-----BEGIN PGP") != std::string::npos) {
      found_pgp_section = true;
#ifdef QD_DEBUG
      std::cout << "Found PGP Section\n";
#endif
    }

    // remove windows file ending ... I hate it ...
    if (line.size() != 0 && line.back() == '\r')
      line.pop_back();

    // new keyword
    if (line[0] == '*' || found_pgp_section) {

      if (!line_buffer.empty() && !last_keyword.empty()) {

        // transfer possible header for following keyword (see function)
        transfer_comment_header(line_buffer, line_buffer_tmp);

        // get type
        auto kw_type = Keyword::determine_keyword_type(last_keyword);

#ifdef QD_DEBUG
        std::cout << last_keyword << " -> ";
        switch (kw_type) {
          case (Keyword::KeywordType::NODE):
            std::cout << "NODE\n";
            break;
          case (Keyword::KeywordType::ELEMENT):
            std::cout << "ELEMENT\n";
            break;
          case (Keyword::KeywordType::PART):
            std::cout << "PART\n";
            break;
          case (Keyword::KeywordType::GENERIC):
            std::cout << "GENERIC\n";
            break;
          case (Keyword::KeywordType::INCLUDE):
            std::cout << "INCLUDE\n";
            break;
          case (Keyword::KeywordType::INCLUDE_PATH):
            std::cout << "INCLUDE_PATH\n";
            break;
        }
#endif

        auto kw = create_keyword(line_buffer,
                                 kw_type,
                                 iLine - line_buffer.size() -
                                   line_buffer_tmp.size() + 1);
        if (kw)
          keywords[kw->get_keyword_name()].push_back(kw);

        // transfer cropped data
        line_buffer = line_buffer_tmp;
      }

      // we always trim keywords
      trim_right(line);
      last_keyword = line;

    } // IF:line[0] == '*'

    // Encrypted Sections
    //
    // Extracts encrypted section here and places it in a line in the
    // line buffer. An encrypted section is treated like a keyword.
    if (found_pgp_section) {
      found_pgp_section = false;

      // get stream position
      const auto stream_position = st.tellg();

      const auto end_position =
        string_buffer.find("-----END PGP", stream_position);

      if (end_position == std::string::npos)
        throw(
          std::runtime_error("Could not find \"-----END PGP MESSAGE-----\" for "
                             "corresponding \"-----BEGIN PGP MESSAGE-----\" "));

      // set stream position behind encrypted section
      st.seekg(end_position);

      // extract encrypted stuff
      line += '\n';
      line += std::string(char_buffer.begin() + stream_position,
                          char_buffer.begin() + end_position);

      // print_string_as_hex(line);

      if (line.back() == '\n')
        line.pop_back();
      if (line.back() == '\r')
        line.pop_back();
    }

    // we stupidly add every line to the buffer
    line_buffer.push_back(line);

  } // for:line

  // allocate last block
  if (!line_buffer.empty() && !last_keyword.empty()) {

    auto kw = create_keyword(line_buffer,
                             Keyword::determine_keyword_type(last_keyword),
                             iLine - line_buffer.size() + 1);
    if (kw)
      keywords[kw->get_keyword_name()].push_back(kw);
  }

  // only load files above *END!
  const auto end_kw_position = get_end_keyword_position();

  // includes
  if (load_includes) {

    // update include dirs
    get_include_dirs(true);

    // do the thing
    for (auto& include_kw : include_keywords) {

      if (include_kw->get_position() < end_kw_position) {

        // Note: prevent loading the mesh here
        include_kw->load(false);
      }
    }
  }

  // wait for threads to finish preloading
  // _wq.wait_for_completion();

  // Wait for completion
  // while (work_queue.size() != 0) {
  //   work_queue.front().wait();
  //   work_queue.pop();
  // }

  // load mesh if requested
  if (parse_mesh && _load_mesh) {

    // load nodes
    load_nodes();

    // load parts
    load_parts();

    // load elements
    load_elements();
  }

  return true;
}

/** Loads the nodes from the keywords into the database
 *
 */
void
KeyFile::load_nodes()
{
  const auto end_kw_position = get_end_keyword_position();

  // load oneself
  for (auto& node_keyword : node_keywords) {
    if (node_keyword->get_position() < end_kw_position)
      node_keyword->load();
  }

  // load includes
  if (load_includes)
    for (auto& include_kw : include_keywords)
      for (auto& include_kf : include_kw->get_includes())
        include_kf->load_nodes();
}

/** Loads the parts from the keywords into the database
 *
 */
void
KeyFile::load_parts()
{
  const auto end_kw_position = get_end_keyword_position();

  // load oneself
  for (auto& part_kw : part_keywords) {
    if (part_kw->get_position() < end_kw_position)
      part_kw->load();
  }

  // load includes
  if (load_includes)
    for (auto& include_kw : include_keywords)
      for (auto& include_kf : include_kw->get_includes())
        include_kf->load_parts();
}

/** Loads the parts from the keywords into the database
 *
 */
void
KeyFile::load_elements()
{
  const auto end_kw_position = get_end_keyword_position();

  // load oneself
  for (auto& element_kw : element_keywords) {
    if (element_kw->get_position() < end_kw_position)
      element_kw->load();
  }

  // load includes
  if (load_includes)
    for (auto& include_kw : include_keywords)
      for (auto& include_kf : include_kw->get_includes())
        include_kf->load_elements();
}

/** Extract a comment header, which may belong to another keyword block
 *
 * @param _old : old string buffer with lines containing coments
 * @param _new : buffer in which the extracted comments will be written
 *
 * The routine does the following:
 * *KEYWORD
 * I'm some data
 * $-------------------------------------
 * $ I should be removed from the upper kw
 * $ because im the header for the lower
 * $-------------------------------------
 * *ANOTHER_KEYWORD
 *
 * If there are comments DIRECTLY ABOVE a keyword, they ought to be transferred
 * to the following one, because people just defined a header. People never
 * define comments at the end of a keyword, and if they do they suck!
 */
void
KeyFile::transfer_comment_header(std::vector<std::string>& _old,
                                 std::vector<std::string>& _new)
{

  size_t nTransferLines = 0;
  if (_old.size() > 0) {
    for (size_t iCount = _old.size() - 1; iCount > 0 && _old[iCount][0] == '$';
         --iCount)
      ++nTransferLines;
  }

  _new.resize(nTransferLines);
  std::copy(_old.end() - nTransferLines, _old.end(), _new.begin());
  //_new = std::vector<std::string>(_old.end() - nTransferLines, _old.end());
  _old.resize(_old.size() - nTransferLines);
}

/** Update the include path
 *
 * @return include filepaths
 *
 * The include path are the directories, in which all includes will be
 * searched for.
 */
const std::vector<std::string>&
KeyFile::get_include_dirs(bool _update)
{

#ifdef QD_DEBUG
  std::cout << "Refreshing include dirs \n";
#endif

  if (!_update)
    return include_dirs;

  // lets take the long way ...
  std::set<std::string> new_include_dirs;

  // dir of file
  std::string directory = "";
  // BUGFIX
  // https://github.com/qd-cae/qd-cae-python/issues/53
  // originally the filepath of the include was used as basis
  // as it seems, dyna uses the master file keyfile directory
  // as basis for any include downwards
  auto master = get_master_keyfile();
  auto my_filepath = master->get_filepath();
  size_t pos = my_filepath.find_last_of("/\\");
  if (pos != std::string::npos)
    directory = my_filepath.substr(0, pos) + "/";

  if (!directory.empty())
    new_include_dirs.insert(directory);

  // update
  for (auto& kw : include_path_keywords) {
    auto kw_inc_path = std::static_pointer_cast<IncludePathKeyword>(kw);
    bool is_relative_dir = kw_inc_path->is_relative();

    // append
    for (const auto& dirpath : kw_inc_path->get_include_dirs()) {

      // BUGFIX + BUGCAUSE
      // we ignore relative or not here and simply search everything
      // this is related to issue:
      // https://github.com/qd-cae/qd-cae-python/issues/53
      new_include_dirs.insert(join_path(directory, dirpath));
      // if (is_relative_dir)
      new_include_dirs.insert(dirpath);
    }
  }

  // check also the includes
  for (auto& include_kw : include_keywords) {
    for (auto& include_kf : include_kw->get_includes()) {
      auto paths = include_kf->get_include_dirs(true);
      new_include_dirs.insert(paths.begin(), paths.end());
    }
  }

  include_dirs =
    std::vector<std::string>(new_include_dirs.begin(), new_include_dirs.end());

#ifdef QD_DEBUG
  std::cout << "Include dirs:\n";
  for (const auto& entry : include_dirs)
    std::cout << entry << '\n';
#endif

  return include_dirs;
}

/** Resolve an include
 *
 * @param _filepath path of the include
 * @return filepath resolved filepath
 */
std::string
KeyFile::resolve_include_filepath(const std::string& _filepath)
{

#ifdef QD_DEBUG
  std::cout << "Resolving " << _filepath << '\n';
#endif

  if (check_ExistanceAndAccess(_filepath))
    return _filepath;

  for (const auto& dir : include_dirs) {
    auto full_path = join_path(dir, _filepath);
#ifdef QD_DEBUG
    std::cout << "Trying " << full_path << '\n';
#endif

    if (check_ExistanceAndAccess(full_path))
      return full_path;
  }

  throw(std::invalid_argument("Can not find include: " + _filepath));
}

/** Get all child include files
 *
 * @return all_includes
 *
 * load_includes must be enabled for includes to be loaded.
 */
std::vector<std::shared_ptr<KeyFile>>
KeyFile::get_includes()
{
  std::vector<std::shared_ptr<KeyFile>> all_includes;

  for (auto& include_kw : include_keywords) {
    auto new_includes = include_kw->get_includes();
    all_includes.insert(
      all_includes.end(), new_includes.begin(), new_includes.end());
  }

  return all_includes;
}

/** Add a keyword from it's definition
 *
 * @param  _lines : lines which define the keyword
 * @return keyword
 *
 * Returns a nullptr if keyword the keyword
 */
std::shared_ptr<Keyword>
KeyFile::add_keyword(const std::vector<std::string>& _lines,
                     int64_t _line_index)
{

  // find keyword name
  const auto it =
    std::find_if(_lines.cbegin(), _lines.cend(), [](const std::string& line) {
      return (line.size() > 0 && line[0] == '*');
    });

  if (it == _lines.cend())
    throw(std::invalid_argument(
      "Can not find keyword definition (line must begin with *)"));

  // determine type
  auto kw_type = Keyword::determine_keyword_type(*it);

  // do the thing
  auto kw = create_keyword(_lines, kw_type, _line_index);
  if (kw)
    keywords[kw->get_keyword_name()].push_back(kw);

  return kw;
}

/** Remove all keywords with the specified name
 *
 * @param _keyword_name
 */
void
KeyFile::remove_keyword(const std::string& _keyword_name)
{
  // search
  auto it = keywords.find(_keyword_name);
  if (it == keywords.end())
    return;

  auto& kwrds = it->second;

  // check for non-generic type
  if (kwrds.size() > 0) {
    auto keyword_type = kwrds[0]->get_keyword_type();
    if (keyword_type != Keyword::KeywordType::GENERIC)
      throw(std::invalid_argument("Can not delete non-generic keywords yet."));
  }

  keywords.erase(it);
}

/** Convert the keyfile to a string
 *
 * @return str : keyfile as string
 */
std::string
KeyFile::str() const
{

  std::vector<std::shared_ptr<Keyword>> kwrds_sorted;
  for (auto& kv : keywords) {
    for (auto kw : kv.second) {
      kwrds_sorted.push_back(kw);
    }
  }

  std::sort(kwrds_sorted.begin(),
            kwrds_sorted.end(),
            [](const std::shared_ptr<Keyword>& instance1,
               const std::shared_ptr<Keyword>& instance2) {
              return instance1->get_position() < instance2->get_position();
            });

  std::stringstream ss;
  for (auto kv : kwrds_sorted)
    ss << kv->str();

  auto str = ss.str();

  if (!has_linebreak_at_eof && str.back() == '\n')
    str.pop_back();

  return std::move(str);
}

/** Save a keyfile again
 *
 * @param _filepath : path to new file
 */
void
KeyFile::save_txt(const std::string& _filepath)
{
  save_file(_filepath, str());
  set_filepath(_filepath);
}

/** Get the position of the lowest *END
 *
 * @return position : position index (usually line)
 *
 * Returns max_position of none found
 */
int64_t
KeyFile::get_end_keyword_position()
{
  int64_t position = max_position;
  for (auto& kw : get_keywordsByName("*END")) {
    position = std::min(position, kw->get_position());
  }
  for (auto& kw : get_keywordsByName("*eND")) {
    position = std::min(position, kw->get_position());
  }
  for (auto& kw : get_keywordsByName("*enD")) {
    position = std::min(position, kw->get_position());
  }
  for (auto& kw : get_keywordsByName("*end")) {
    position = std::min(position, kw->get_position());
  }
  for (auto& kw : get_keywordsByName("*End")) {
    position = std::min(position, kw->get_position());
  }
  for (auto& kw : get_keywordsByName("*EnD")) {
    position = std::min(position, kw->get_position());
  }
  for (auto& kw : get_keywordsByName("*ENd")) {
    position = std::min(position, kw->get_position());
  }
  for (auto& kw : get_keywordsByName("*eNd")) {
    position = std::min(position, kw->get_position());
  }
  return position;
}

} // namespace qd