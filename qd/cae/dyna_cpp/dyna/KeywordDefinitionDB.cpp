
#include <dyna_cpp/dyna/KeywordDefinitionDB.hpp>

namespace qd {

/** Constructor for the KeywordDefinitionDB
 *
 * Does nothing.
 */
KeywordDefinitionDB::KeywordDefinitionDB() {}

/** Add a keyword definition to the database
 *
 * @param _keyword_name name of the new keyword definition
 * @return _
 */
std::shared_ptr<KeywordDefinition>
KeywordDefinitionDB::add_keyword_definition(const std::string& _keyword_name)
{
  // check previous definition
  auto it = keyword_defs.find(_keyword_name);
  if (it != keyword_defs.end())
    throw(std::invalid_argument(
      "A keyword definition with the following name does already exist:" +
      _keyword_name));

  // insert new definition
  auto keyword_def = std::make_shared<KeywordDefinition>(_keyword_name);
  auto pair = std::pair<std::string, std::shared_ptr<KeywordDefinition>>(
    _keyword_name, keyword_def);
  keyword_defs.insert(pair);

  // return
  return keyword_def;
}

/** Erase a keyword definition by its name
 *
 * @param _keyword_name name of the keyword to erase
 *
 * If there is no such keyword definition in the DB, nothing happens
 */
void
KeywordDefinitionDB::remove_keyword_definition(const std::string& _keyword_name)
{
  auto it = keyword_defs.find(_keyword_name);
  if (it != keyword_defs.end())
    keyword_defs.erase(it);
}

} // namespace qd