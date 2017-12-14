
#ifndef KEYWORDDEFINITIONDB_HPP
#define KEYWORDDEFINITIONDB_HPP

// includes
#include <dyna_cpp/dyna/Keyword.hpp>
#include <dyna_cpp/dyna/KeywordDefinition.hpp>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace qd {

class KeywordDefinitionDB
{
private:
  std::unordered_map<std::string, std::shared_ptr<KeywordDefinition>>
    keyword_defs;
  KeywordDefinitionDB();

public:
  static KeywordDefinitionDB& instance();
  std::shared_ptr<KeywordDefinition> add_keyword_definition(
    const std::string& _keyword_name);
  void remove_keyword_definition(const std::string& _keyword_name);

  std::vector<std::string> get_keyword_definition_names();
  std::vector<std::shared_ptr<KeywordDefinition>> get_keyword_definitions(
    const std::string& _keyword_name);
};

/** Get the unique
 *
 */
KeywordDefinitionDB&
KeywordDefinitionDB::instance()
{
  static KeywordDefinitionDB unique_instance;
  return unique_instance;
}

} // namespace qd

#endif