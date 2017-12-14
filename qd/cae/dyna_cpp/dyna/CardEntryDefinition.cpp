
#include <dyna_cpp/dyna/CardEntryDefinition.hpp>

namespace qd {

/** Constructor for the definition of a keyword card entry ... blabla
 *
 * @param _index index of the entry (which column)
 * @param _char_width length of the entry in characters
 * @param _var_type type of the variable to parse
 * @param _card_name name of the entry to display
 */
CardEntryDefinition::CardEntryDefinition(int64_t _char_start,
                                         int64_t _char_width,
                                         VariableType _var_type,
                                         const std::string& _card_entry_name,
                                         const std::string& _default_value)
  : card_entry_name(_card_entry_name)
  , var_type(_var_type)
{

  // setters with checks
  set_char_start(_char_start);
  set_char_width(_char_width);
  // set_variable_type(_var_type);
  set_default_value(_default_value);
}

/** Check if a string is convertible in the specified type
 *
 * @param _string string to check
 * @param _var_type type to convert into
 */
bool
CardEntryDefinition::is_convertible(const std::string& _string,
                                    VariableType _var_type)
{

  // float
  if (_var_type == VariableType::FLOAT32 ||
      _var_type == VariableType::FLOAT64) {
    try {
      std::stol(this->default_value);
      return true;
    } catch (...) {
      return false;
    }

    // integral
  } else if (_var_type == VariableType::INT32 ||
             _var_type == VariableType::INT64) {
    try {
      std::stod(this->default_value);
      return true;
    } catch (...) {
      return false;
    }
  }
  // string & none
  return true;
}

/** Set the starting index of the card entry (chars)
 *
 * @param _start starting index, bounded by [0,79]
 */
void
CardEntryDefinition::set_char_start(int64_t _start)
{
  // checks
  if (_start < 0)
    throw(
      std::invalid_argument("character starting index may not be negative."));
  if (_start > 79)
    throw(std::invalid_argument("character starting index too large, LS-Dyna "
                                "only allows 80 characters per line."));

  this->char_start = _start;
}

/** Set the char width of card entry
 *
 * @param _start starting index, bounded by [0,79]
 */
void
CardEntryDefinition::set_char_width(int64_t _width)
{
  // checks
  if (_width < 0)
    throw(std::invalid_argument("character width may not be negative."));
  if (char_start + _width > 79)
    throw(std::invalid_argument("card entry size too large, LS-Dyna "
                                "only allows 80 characters per line. Either "
                                "reduce the specified width or the starting "
                                "character."));

  this->char_width = _width;
}

/** Set the name of the card
 *
 * @param _name name of the card
 */
void
CardEntryDefinition::set_card_entry_name(const std::string _name)
{
  this->card_entry_name = _name;
}

/** Set the type of the card variable
 *
 * @param _type type of the element (CardEntryDefinition::VariableType)
 */
void
CardEntryDefinition::set_variable_type(VariableType _type)
{

  // check conversion
  if (!default_value.empty() && !is_convertible(default_value, _type)) {
    throw(
      std::invalid_argument("Can not convert default value:" + default_value +
                            " to the specified variable type."));
  }

  var_type = _type;
}

/** Set a new default value for the entry
 *
 * @param _value new default value
 */
void
CardEntryDefinition::set_default_value(const std::string& _value)
{

  // check conversion
  if (!_value.empty() && var_type != VariableType::NONE) {
    throw(
      std::invalid_argument("Can not convert default value:" + default_value +
                            " to the specified variable type."));
  }
}

/** Get the starting index of the characters
 *
 * @return index
 */
int64_t
CardEntryDefinition::get_char_start() const
{
  return char_start;
}

/** Get the entry width in characters
 *
 * @return index
 */
int64_t
CardEntryDefinition::get_char_width() const
{
  return char_width;
}

/** Get the name of the card
 *
 * @return name
 */
std::string
CardEntryDefinition::get_card_entry_name() const
{
  return card_entry_name;
}

/** Get the type of the variable
 *
 * @return variable type
 */
CardEntryDefinition::VariableType
CardEntryDefinition::get_variable_type() const
{
  return var_type;
}

/** Get the default value
 *
 * @return default value
 */
std::string
CardEntryDefinition::get_default_value() const
{
  return default_value;
}

/** Validate the uniqueness against another definition
 *
 * @param _def other card entry definition
 */
void
CardEntryDefinition::validate_uniqueness(
  std::shared_ptr<CardEntryDefinition> _def)
{

  // check unique name
  if (this->get_card_entry_name() == _def->get_card_entry_name())
    throw(std::invalid_argument("Card entry with name " +
                                this->get_card_entry_name() +
                                " does already exist in CardDefinition."));
  // check char indexes
  if (((_def->get_char_start() > this->get_char_start()) &&
       (_def->get_char_start() <
        this->get_char_start() + this->get_char_width)) ||
      ((_def->get_char_start() + _def->get_char_width() >
        this->get_char_start()) &&
       (_def->get_char_start() + _def->get_char_width() <
        this->get_char_start() + this->get_char_width)))

    throw(std::invalid_argument(
      "Card entry:" + _def->get_card_entry_name() +
      " would be inside the previously defined card entry:" +
      this->get_card_entry_name()));
}

} // namespace qd