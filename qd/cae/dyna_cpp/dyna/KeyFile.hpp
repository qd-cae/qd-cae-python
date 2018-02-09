
#ifndef KEYFILE_HPP
#define KEYFILE_HPP

// includes
#include <dyna_cpp/db/FEMFile.hpp>
#include <dyna_cpp/dyna/ElementKeyword.hpp>
#include <dyna_cpp/dyna/IncludeKeyword.hpp>
#include <dyna_cpp/dyna/IncludePathKeyword.hpp>
#include <dyna_cpp/dyna/Keyword.hpp>
#include <dyna_cpp/dyna/NodeKeyword.hpp>
#include <dyna_cpp/dyna/PartKeyword.hpp>

#include <map>
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

  double encryption_detection_threshold;
  std::vector<std::string> include_dirs;

  std::map<std::string, std::vector<std::shared_ptr<Keyword>>> keywords;

  // internal use only
  std::vector<std::shared_ptr<NodeKeyword>> node_keywords;
  std::vector<std::shared_ptr<ElementKeyword>> element_keywords;
  std::vector<std::shared_ptr<PartKeyword>> part_keywords;
  std::vector<std::shared_ptr<IncludeKeyword>> include_keywords;
  std::vector<std::shared_ptr<IncludePathKeyword>> include_path_keywords;

  std::shared_ptr<Keyword> create_keyword(
    const std::vector<std::string>& _lines,
    Keyword::KeywordType _keyword_type,
    size_t _iLine,
    bool _insert_into_buffer);

  void transfer_comment_header(std::vector<std::string>& _old,
                               std::vector<std::string>& _new);
  const std::vector<std::string>& get_include_dirs(bool _update = false);
  void update_keyword_names(); // TODO

  void load_include_files();
  void load_nodes();
  void load_parts();
  void load_elements();

  std::vector<std::shared_ptr<Keyword>> get_keywordsByType(
    Keyword::KeywordType _type);

public:
  KeyFile(bool _read_generic_keywords = false,
          bool _parse_mesh = false,
          bool _load_includes = true,
          double _encryption_detection = 0.7,
          KeyFile* _parent_kf = nullptr);
  KeyFile(const std::string& _filepath,
          bool _read_generic_keywords = false,
          bool _parse_mesh = false,
          bool _load_includes = true,
          double _encryption_detection = 0.7,
          KeyFile* _parent_kf = nullptr);

  void load(bool _load_mesh = false);
  inline std::vector<std::shared_ptr<Keyword>> get_keywordsByName(
    const std::string& _keyword_name);
  template<typename T>
  void remove_keyword(const std::string& _keyword_name, T _index);
  inline std::vector<std::string> keys();
  std::string str() const;
  void save_txt(const std::string& _filepath,
                bool _save_includes = true,
                bool _save_all_in_one = false);

  std::string resolve_include_filepath(const std::string& _filepath);

  bool get_read_generic_keywords() const { return read_generic_keywords; }
  bool get_parse_mesh() const { return parse_mesh; }
  bool get_load_includes() const { return load_includes; }
  double get_encryption_detection_threshold() const
  {
    return encryption_detection_threshold;
  }
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

} // namespace qd

#endif
