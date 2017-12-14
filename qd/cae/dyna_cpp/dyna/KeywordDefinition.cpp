
#include <stdexcept>

#include <absl/strings/str_join.h>
#include <dyna_cpp/dyna/KeywordDefinition.hpp>

namespace qd {

/** Constructor for a KeywordDefinition
 *
 * @param _name name of the keyword
 */
KeywordDefinition::KeywordDefinition(const std::string& _name)
  : keyword_name(_name)
{}

/** Add a card definition
 *
 * @param _card_def definition of a card in the keyword
 */
void
KeywordDefinition::add_card_definition(
  std::shared_ptr<CardDefinition> _card_def)
{
  this->card_defs.push_back(_card_def);
}

/** Get a card of the keyword by its index
 *
 * @param _iCard index of the card (starts with 0)
 * @return card definition of the card
 */
std::shared_ptr<CardDefinition>
KeywordDefinition::get_card_definition(int64_t _iCard)
{
  if (_iCard < 0)
    throw(std::invalid_argument(
      "Can not get a card from a keyword with negative index: " +
      std::to_string(_iCard)));
}

/** Get the number of cards
 *
 *  @return nCards numbe of cards in the keyword
 */
int64_t
KeywordDefinition::get_nCards() const
{
  return static_cast<int64_t>(card_defs.size());
}

} // namespace qd