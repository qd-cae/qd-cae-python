
#include <dyna_cpp/dyna/Keyword.hpp>

namespace qd {

/** Construct a keyword from a definition
 *
 * @param _definition definition of the Keyword.
 */
Keyword::Keyword(std::shared_ptr<KeywordDefinition> _definition) {}

/** Get the name of the keyword
 *
 * @return name name of the keyword
 */
std::string
Keyword::get_keyword_name() const
{
  return this->definition->get_keyword_name();
}

} // namespace qd