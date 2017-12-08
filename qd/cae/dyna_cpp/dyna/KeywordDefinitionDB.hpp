
#ifndef KEYWORDDEFINITIONDB_HPP
#define KEYWORDDEFINITIONDB_HPP

// includes
#include <dyna_cpp/dyna/Keyword.hpp>
#include <dyna_cpp/dyna/KeywordDefinition.hpp>

#include <unordered_map>
#include <memory>
#include <string>
#include <vector>

namespace qd {

class KeywordDefinitionDB
{
private:
  static std::shared_ptr<KeywordDefinitionDB> instance;
  std::unordered_map<std::string, std::shared_ptr<KeywordDefinition>> keyword_defs;
  KeywordDefinitionDB();

public:
  static std::shared_ptr<KeywordDefinitionDB> instance();
  void add_keyword_definition(std::shared_ptr<KeywordDefinition> _definition);
  void remove_keyword_definition(const std::string& _name);

  std::vector<std::shared_ptr<KeywordDefinition>> get_keyword_definitions(
    const std::string& _keyword_name);
};

} // namespace qd

#endif