
#ifndef KEYFILE_HPP
#define KEYFILE_HPP

// includes
#include <dyna_cpp/db/FEMFile.hpp>
#include <dyna_cpp/dyna/Keyword.hpp>

#include <map>
#include <stdexcept>
#include <string>

// forward declarations
class D3plot;

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
    ELEMENT_BEAM,
    ELEMENT_SHELL,
    ELEMENT_SOLID,
    PART,
    INCLUDE,
    COMMENT,
    GENERIC
  };

private:
  bool load_includes;
  double encryption_detection_threshold;
  std::vector<std::string> base_dirs;
  std::vector<std::shared_ptr<KeyFile>> includes;
  std::map<std::string, std::vector<std::shared_ptr<Keyword>>> keywords;

  void read_mesh(const std::string& _filepath);
  void parse_file(const std::string& _filepath);
  void create_keyword(const std::vector<std::string>& _lines,
                      const std::string& _keyword_name,
                      size_t _iLine);
  std::string resolve_include(const std::string& _filepath);

public:
  KeyFile();
  KeyFile(const std::string& _filepath,
          bool _load_includes = true,
          double _encryption_detection = 0.7);
  inline std::vector<std::shared_ptr<Keyword>> get_keywordsByName(
    const std::string& _keyword_name);
  inline std::vector<std::string> keys();
  void save_keyfile(const std::string& _filepath,
                    bool _save_includes = true,
                    bool _save_all_in_one = false);
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
