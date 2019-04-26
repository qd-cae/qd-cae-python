
#ifndef KEYFILE_HPP
#define KEYFILE_HPP

// includes
#include <dyna_cpp/db/FEMFile.hpp>
#include <dyna_cpp/dyna/keyfile/ElementKeyword.hpp>
#include <dyna_cpp/dyna/keyfile/IncludeKeyword.hpp>
#include <dyna_cpp/dyna/keyfile/IncludePathKeyword.hpp>
#include <dyna_cpp/dyna/keyfile/Keyword.hpp>
#include <dyna_cpp/dyna/keyfile/NodeKeyword.hpp>
#include <dyna_cpp/dyna/keyfile/PartKeyword.hpp>
// #include <dyna_cpp/parallel/WorkQueue.hpp>

#include <map>
#include <queue>
#include <stdexcept>
#include <string>

namespace qd {

/**
 * This is a class for reading LS-Dyna input files.The mesh will be parsed with
 * it's properties, currently only in a limited way.
 */
class KeyFile : public FEMFile
{
public:
  enum class KeywordType
  {
    NONE,
    NODE,
    ELEMENT,
    PART,
    INCLUDE,
    COMMENT,
    GENERIC
  };

private:
  KeyFile* parent_kf;
  bool load_includes;
  bool read_generic_keywords;
  bool parse_mesh;
  bool has_linebreak_at_eof;
  int64_t max_position;

  std::vector<std::string> include_dirs;
  std::map<std::string, std::vector<std::shared_ptr<Keyword>>> keywords;

  // internal use only
  std::vector<std::shared_ptr<NodeKeyword>> node_keywords;
  std::vector<std::shared_ptr<ElementKeyword>> element_keywords;
  std::vector<std::shared_ptr<PartKeyword>> part_keywords;
  std::vector<std::shared_ptr<IncludeKeyword>> include_keywords;
  std::vector<std::shared_ptr<IncludePathKeyword>> include_path_keywords;

  // parallel worker queue
  //  WorkQueue _wq;
  // std::queue<std::future<void>> work_queue;

  template<typename T>
  std::shared_ptr<Keyword> create_keyword(
    const std::vector<std::string>& _lines,
    Keyword::KeywordType _keyword_type,
    T _iLine);

  void transfer_comment_header(std::vector<std::string>& _old,
                               std::vector<std::string>& _new);
  void update_keyword_names(); // TODO

  void load_nodes();
  void load_parts();
  void load_elements();

public:
  KeyFile(bool _read_generic_keywords = false,
          bool _parse_mesh = false,
          bool _load_includes = true,
          KeyFile* _parent_kf = nullptr);
  KeyFile(const std::string& _filepath,
          bool _read_generic_keywords = false,
          bool _parse_mesh = false,
          bool _load_includes = true,
          KeyFile* _parent_kf = nullptr);

  size_t get_nTimesteps() const override { return 1; };

  bool load(bool _load_mesh = true);

  // keywords
  inline std::vector<std::string> keys();
  inline std::vector<std::shared_ptr<Keyword>> get_keywordsByName(
    const std::string& _keyword_name);
  std::shared_ptr<Keyword> add_keyword(const std::vector<std::string>& _lines,
                                       int64_t _line_index = 0);
  void remove_keyword(const std::string& _keyword_name);
  template<typename T>
  void remove_keyword(const std::string& _keyword_name, T _index);
  int64_t get_end_keyword_position();

  // io
  std::string str() const;
  void save_txt(const std::string& _filepath);
  std::string resolve_include_filepath(const std::string& _filepath);
  std::vector<std::shared_ptr<KeyFile>> get_includes();
  const std::vector<std::string>& get_include_dirs(bool _update = false);

  bool get_read_generic_keywords() const { return read_generic_keywords; }
  bool get_parse_mesh() const { return parse_mesh; }
  bool get_load_includes() const { return load_includes; }

  // utility
  inline KeyFile* get_master_keyfile();
};

/** Get all keywords with a specific name
 *
 * @param _keyword_name name of the keyword
 * @return keywords vector of keywords
 *
 * Returns an empty vector if none where found.
 */
std::vector<std::shared_ptr<Keyword>>
KeyFile::get_keywordsByName(const std::string& _keyword_name)
{

  // search and return
  auto it = keywords.find(_keyword_name);
  if (it != keywords.end())
    return it->second;

  // return empty vector if not found
  // (prevents creation of an empty entry in map)
  return std::vector<std::shared_ptr<Keyword>>();
}

/** Remove a keyword
 *
 * @param _keyword_name : name
 * @param _index : index in the vector
 *
 * Does nothing if keyword does not exist or
 */
template<typename T>
void
KeyFile::remove_keyword(const std::string& _keyword_name, T _index)
{
  // search
  auto it = keywords.find(_keyword_name);
  if (it == keywords.end())
    return;

  auto& kwrds = it->second;

  // do the thing
  _index = index_treatment(_index, kwrds.size());

  if (static_cast<size_t>(_index) < kwrds.size() &&
      kwrds[_index]->get_keyword_type() == Keyword::KeywordType::GENERIC)
    kwrds.erase(kwrds.begin() + _index);

  // remove keyword name in map if no data is here
  if (kwrds.size() == 0)
    keywords.erase(it);

  if (kwrds[_index]->get_keyword_type() != Keyword::KeywordType::GENERIC)
    throw(std::invalid_argument("Can not delete non-generic keywords yet."));
}

/** Get a list of keywords in the file
 *
 * @return list list of keyword names loaded
 */
std::vector<std::string>
KeyFile::keys()
{
  std::vector<std::string> list;
  for (const auto& kv : keywords)
    list.push_back(kv.first);
  return list;
}

/** Get the uppermost keyfile (owning the mesh)
 *
 * @return master
 */
KeyFile*
KeyFile::get_master_keyfile()
{
  if (parent_kf == this) {
    return this;
  } else
    return parent_kf->get_master_keyfile();
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
template<typename T>
std::shared_ptr<Keyword>
KeyFile::create_keyword(const std::vector<std::string>& _lines,
                        Keyword::KeywordType _keyword_type,
                        T _iLine)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  if (_iLine < 0)
    _iLine = max_position + 1;
  auto position = static_cast<int64_t>(_iLine);

  if (parse_mesh) {

    switch (_keyword_type) {
      case (Keyword::KeywordType::NODE): {
        auto kw = std::make_shared<NodeKeyword>(
          get_master_keyfile()->get_db_nodes(), _lines, position);
        node_keywords.push_back(kw);

        // preload while continuing parsing
        // parent_kf->work_queue.push(std::async([kw]() { kw->load(); }));

        // update max position
        max_position = std::max(position, max_position);

        return kw;
        break;
      }
      case (Keyword::KeywordType::ELEMENT): {
        auto kw = std::make_shared<ElementKeyword>(
          get_master_keyfile()->get_db_elements(),
          _lines,
          static_cast<int64_t>(_iLine));
        element_keywords.push_back(kw);

        // update max position
        max_position = std::max(position, max_position);

        return kw;
        break;
      }
      case (Keyword::KeywordType::PART): {
        auto kw =
          std::make_shared<PartKeyword>(get_master_keyfile()->get_db_parts(),
                                        _lines,
                                        static_cast<int64_t>(_iLine));
        part_keywords.push_back(kw);

        // update max position
        max_position = std::max(position, max_position);

        // _wq.submit([](std::shared_ptr<PartKeyword> kw) { kw->load(); }, kw);
        return kw;
        break;
      }
      default:
        // nothing
        break;
    }
  }

  if (load_includes) {

    // *INCLUDE_PATH
    if (_keyword_type == Keyword::KeywordType::INCLUDE_PATH) {
      auto kw = std::make_shared<IncludePathKeyword>(
        _lines, static_cast<int64_t>(_iLine));
      include_path_keywords.push_back(kw);

      // update max position
      max_position = std::max(position, max_position);

      return kw;
    }

    // *INCLUDE
    if (_keyword_type == Keyword::KeywordType::INCLUDE) {
      auto kw = std::make_shared<IncludeKeyword>(
        this, _lines, static_cast<int64_t>(_iLine));
      include_keywords.push_back(kw);

      // update max position
      max_position = std::max(position, max_position);

      return kw;
    }
  }

  if (read_generic_keywords) {

    // update max position
    max_position = std::max(position, max_position);
    return std::make_shared<Keyword>(_lines, _iLine);

  } else
    return nullptr;
}

} // namespace qd

#endif
