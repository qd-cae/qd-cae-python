
#include <cstdint>
#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>

#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>
#include <dyna_cpp/db/Element.hpp>
#include <dyna_cpp/db/Part.hpp>
#include <dyna_cpp/dyna/ElementKeyword.hpp>
#include <dyna_cpp/dyna/IncludePathKeyword.hpp>
#include <dyna_cpp/dyna/KeyFile.hpp>
#include <dyna_cpp/dyna/NodeKeyword.hpp>
#include <dyna_cpp/dyna/PartKeyword.hpp>
#include <dyna_cpp/utility/FileUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

namespace qd {

/**
 * Constructor for a LS-Dyna input file.
 */
KeyFile::KeyFile(bool _read_generic_keywords,
                 bool _parse_mesh,
                 bool _load_includes,
                 double _encryption_detection,
                 KeyFile* _parent_kf)
  : KeyFile("", _parse_mesh, _load_includes, _encryption_detection, _parent_kf)
{}

/** Constructor for reading a LS-Dyna input file.
 *
 * @param string filepath : filepath of a key file to read
 */
KeyFile::KeyFile(const std::string& _filepath,
                 bool _read_generic_keywords,
                 bool _parse_mesh,
                 bool _load_includes,
                 double _encryption_detection,
                 KeyFile* _parent_kf)
  : FEMFile(_filepath)
  , parent_kf(_parent_kf)
  , load_includes(_load_includes)
  , read_generic_keywords(_read_generic_keywords)
  , parse_mesh(_parse_mesh)
  , encryption_detection_threshold(_encryption_detection)
{
  if (parent_kf == nullptr)
    parent_kf = this;

  // check encryption
  if (encryption_detection_threshold < 0 || encryption_detection_threshold > 1)
    throw(std::invalid_argument(
      "Encryption detection threshold must be between 0 and 1."));
}

/** Parse a keyfile
 *
 * @param _load_mesh : whether the mesh shall loaded
 *
 * The parameter can be used to prevent the loading of the mesh,
 * even though we use parse_mesh. We need this for includes.
 */
void
KeyFile::load(bool _load_mesh)
{

  // read file
  auto my_filepath = resolve_include_filepath(get_filepath());
  std::vector<char> char_buffer = read_binary_file(my_filepath);
#ifdef QD_DEBUG
  std::cout << "done." << std::endl;
#endif

  // test for encryption
  if ((get_entropy(char_buffer) / 8.) > this->encryption_detection_threshold) {
#ifdef QD_DEBUG
    std::cout << "Skipping file " << my_filepath
              << " with normalized entropy of "
              << (get_entropy(char_buffer) / 8) << std::endl;
#endif
    return;
  }

  // convert buffer into blocks
  size_t iLine = 0;
  std::string last_keyword;
  std::vector<std::string> line_buffer;
  std::vector<std::string> line_buffer_tmp;
  std::queue<size_t> buffer_iLine_queue;

  std::string line;
  std::stringstream st(std::string(char_buffer.begin(), char_buffer.end()));
  for (; std::getline(st, line); ++iLine) {

    // remove windows file ending ... I hate it ...
    if (line.size() != 0)
      if (line[line.size() - 1] == '\r')
        line.pop_back();

    // new keyword
    if (line[0] == '*') {

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

    // we stupidly add every line to the buffer
    line_buffer.push_back(line);

  } // for:line

  // allocate last block
  if (!line_buffer.empty() && !last_keyword.empty()) {

    auto kw_type = Keyword::determine_keyword_type(last_keyword);

    auto kw = create_keyword(line_buffer,
                             Keyword::determine_keyword_type(last_keyword),
                             iLine - line_buffer.size() + 1);
    if (kw)
      keywords[kw->get_keyword_name()].push_back(kw);
  }

  // includes
  if (load_includes) {

    // update include dirs
    get_include_dirs(true);

    // do the thing
    for (auto& include_kw : include_keywords) {
      include_kw->load(false); // prevent loading the mesh here
    }
  }

  // load mesh if requested
  if (parse_mesh && _load_mesh) {
    // load nodes
    load_nodes();

    // load parts
    load_parts();

    // load elements
    load_elements();
  }
}

/** Loads the nodes from the keywords into the database
 *
 */
void
KeyFile::load_nodes()
{

  // load oneself
  for (auto& node_keyword : node_keywords) {
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

  // load oneself
  for (auto& part_kw : part_keywords) {
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

  // load oneself
  for (auto& element_kw : element_keywords) {
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

  size_t iCount = 0;
  size_t nTransferLines = 0;
  if (_old.size() > 0)
    for (size_t iCount = _old.size() - 1; iCount > 0 && _old[iCount][0] == '$';
         --iCount)
      ++nTransferLines;

  _new.resize(nTransferLines);
  std::copy(_old.end() - nTransferLines, _old.end(), _new.begin());
  //_new = std::vector<std::string>(_old.end() - nTransferLines, _old.end());
  _old.resize(_old.size() - nTransferLines);
}

/** Create a keyword from it's line buffer
 *
 * @param _lines : buffer
 * @param _keyword_type : type if the keyword
 * @param _iLine : line index of the block
 * @param _insert_into_buffer : whether to insert typed keywords into their
 * buffers
 *
 * The *typed* keywords not inserted into the loading buffers are not loaded
 * automatically and require to call the "load" function
 */
std::shared_ptr<Keyword>
KeyFile::create_keyword(const std::vector<std::string>& _lines,
                        Keyword::KeywordType _keyword_type,
                        size_t _iLine)
{

  if (parse_mesh) {

    switch (_keyword_type) {
      case (Keyword::KeywordType::NODE): {
        auto kw = std::make_shared<NodeKeyword>(
          parent_kf->get_db_nodes(), _lines, static_cast<int64_t>(_iLine));
        node_keywords.push_back(kw);
        return kw;
        break;
      }
      case (Keyword::KeywordType::ELEMENT): {
        auto kw = std::make_shared<ElementKeyword>(
          parent_kf->get_db_elements(), _lines, static_cast<int64_t>(_iLine));
        element_keywords.push_back(kw);
        return kw;
        break;
      }
      case (Keyword::KeywordType::PART): {
        auto kw = std::make_shared<PartKeyword>(
          parent_kf->get_db_parts(), _lines, static_cast<int64_t>(_iLine));
        part_keywords.push_back(kw);
        return kw;
        break;
      }
      default:
        // nothing
        break;
    }
  }

  // *INCLUDE_PATH
  if (_keyword_type == Keyword::KeywordType::INCLUDE_PATH) {
    auto kw = std::make_shared<IncludePathKeyword>(
      _lines, static_cast<int64_t>(_iLine));
    include_path_keywords.push_back(kw);
    return kw;
  }

  // *INCLUDE
  if (_keyword_type == Keyword::KeywordType::INCLUDE) {
    auto kw = std::make_shared<IncludeKeyword>(
      parent_kf, _lines, static_cast<int64_t>(_iLine));
    include_keywords.push_back(kw);
    return kw;
  }

  if (read_generic_keywords)
    return std::make_shared<Keyword>(_lines, _iLine);
  else
    return nullptr;
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

  if (!_update)
    return include_dirs;

  // lets take the long way ...
  std::set<std::string> new_include_dirs;

  // dir of file
  std::string directory = "";
  auto my_filepath = get_filepath();
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
      if (is_relative_dir)
        new_include_dirs.insert(join_path(directory, dirpath));
      else
        new_include_dirs.insert(dirpath);
    }
  }

  // check also the includes
  for (auto& include_kw : include_keywords)
    for (auto& include_kf : include_kw->get_includes()) {
      auto paths = include_kf->get_include_dirs(true);
      new_include_dirs.insert(paths.begin(), paths.end());
    }

#ifdef QD_DEBUG
  std::cout << "Include dirs:\n";
  for (const auto& entry : include_dirs)
    std::cout << entry << '\n';
#endif

  include_dirs =
    std::vector<std::string>(new_include_dirs.begin(), new_include_dirs.end());

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

  if (check_ExistanceAndAccess(_filepath))
    return _filepath;

  for (const auto& dir : include_dirs) {
    if (check_ExistanceAndAccess(dir + _filepath))
      return dir + _filepath;
  }

  throw(std::runtime_error("Can not find anywhere:" + _filepath));
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

  std::map<int64_t, std::shared_ptr<Keyword>> kwrds_sorted;
  for (auto& kv : keywords) {
    for (auto kw : kv.second) {
      kwrds_sorted.insert(std::make_pair(kw->get_line_index(), kw));
    }
  }

  std::stringstream ss;
  for (auto kv : kwrds_sorted)
    ss << kv.second->str();

  return std::move(ss.str());
}

/** Save a keyfile again
 *
 * @param _filepath : path to new file
 */
void
KeyFile::save_txt(const std::string& _filepath)
{
  save_file(_filepath, str());
}

} // namespace qd