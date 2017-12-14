
#include "dyna_cpp/dyna//CardDefinition.hpp"
#include <dyna_cpp/utility/MathUtility.hpp>

namespace qd {

/** Constructor for a Card Definition
 *
 */
CardDefinition::CardDefinition(std::shared_ptr<KeywordDefinition> _parent,
                               bool _is_optional)
  : is_optional(_is_optional)
  , parent_keyword(_parent)
{}

/** Add a card entry definition to a card
 *
 * @param _entry the new card entry
 *
 * Will be checked by name and also overlapping char indexes.
 */
void
CardDefinition::add_card_entry_definition(
  std::shared_ptr<CardEntryDefinition> _entry)
{
  for (auto def : card_entry_defs)
    def->validate_uniqueness(_entry);

  card_entry_defs.push_back(_entry);
}

/** Get a card entry definition
 *
 * @param _name name of the card entry
 * @return card card entry definition
 */
std::shared_ptr<CardEntryDefinition>
CardDefinition::get_card_entry_definition(std::string _name)
{
  for (auto entry : card_entry_defs)
    if (entry->get_card_entry_name() == _name)
      return entry;
  throw(std::invalid_argument("Can not find card entry definition with name:" +
                              _name));
}

/** Get a card entry definition
 *
 * @param _index index of the card entry
 * @return card card entry definition
 */
std::shared_ptr<CardEntryDefinition>
CardDefinition::get_card_entry_definition(int64_t _index)
{
  _index = index_treatment(_index, card_entry_defs.size());
  return card_entry_defs[_index];
}

} // namespace qd