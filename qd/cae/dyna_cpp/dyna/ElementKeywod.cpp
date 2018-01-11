
#include <iomanip>

#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/dyna/ElementKeyword.hpp>

namespace qd {

/** Constructor of an ElementKeyword
 * @param _db_elems : parent database
 * @param _lines : lines of the keyword
 * @param_iLine : line index of keyword
 */
ElementKeyword::ElementKeyword(DB_Elements* _db_elems,
                               const std::string& _lines,
                               int64_t _iLine)
  : Keyword(_lines, _iLine)
  , db_elems(_db_elems)
  , type(Element::ElementType::NONE)
{

  // prepare extraction
  field_size = has_long_fields() ? 16 : 8;

  // what element type do we have?
  auto keyword_name = to_lower_copy(get_keyword_name());
  type = determine_elementType(keyword_name);
  if (type == Element::ElementType::NONE)
    throw(
      std::invalid_argument("Can not find out, what type of element "
                            "(beam,shell,solid,tshell) the keyword contains:"));

  // do the thing
  if (type == Element::ElementType::BEAM)
    parse_elem2(keyword_name, lines);
  if (type == Element::ElementType::SHELL)
    ; // TODO SHELL PARSING
  if (type == Element::ElementType::SOLID)
    ; // TODO SOLID PARSING
  if (type == Element::ElementType::TSHELL)
    ; // TODO TSHELL PARSING
}

/** Parse the string buffer as beam element
 *
 * @param _keyword_name_lower : keyword name in lowercase
 * @param _lines : lines to parse
 */
void
ElementKeyword::parse_elem2(const std::string& _keyword_name_lower,
                            const std::vector<std::string>& _lines)
{

  // find first card line
  size_t header_size = iCard_to_iLine(0, false);
  size_t iLine = header_size;

  // how much data to read
  size_t nAdditionalLines = 0;
  if (_keyword_name_lower.find("thickness") != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("section") != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("scalar") != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("scalr") != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("pid") != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("offset") != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("orientation") != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("warpage") != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("elbow") != std::string::npos)
    ++nAdditionalLines;

  // extract node data
  int32_t element_id;
  int32_t part_id;
  std::vector<int32_t> node_ids(2);
  for (; iLine < lines.size(); iLine += 1 + nAdditionalLines) {

    const auto& line = lines[iLine];
    auto line_trimmed = trim_copy(line);

    if (line_trimmed.empty() || is_keyword(line) || is_comment(line))
      break;

    // parse line
    try {
      element_id = std::stoi(line.substr(0, field_size));
      part_id = std::stoi(line.substr(field_size, field_size));
      node_ids[0] = std::stoi(line.substr(2 * field_size, field_size));
      node_ids[1] = std::stoi(line.substr(3 * field_size, field_size));

      // save remaining element data
      std::string remaining_data(line.begin() + 4 * field_size, line.end());
      for (size_t iExtraLine = 0; iExtraLine < nAdditionalLines; ++iExtraLine)
        remaining_data += '\n' + lines[iLine + 1 + iExtraLine];

      add_elementByNodeID(element_id, part_id, node_ids);
      elem_part_ids.push_back(part_id);
      unparsed_element_data.push_back(remaining_data);

    } catch (const std::out_of_range& err) {
      std::cout << "Parsing error in line: " << (_iLine + iLine + 1) << '\n'
                << "error:" << err.what() << '\n'
                << "line :" << line << '\n';
    }
  }

  // push the gear in the rear :P
  for (; iLine < lines.size(); ++iLine)
    trailing_lines.push_back(lines[iLine]);

  // remove all lines below keyword
  lines.resize(header_size);
}

/** Get the element type from the keyword name
 *
 * @param _keyword_name
 * @return type
 */
Element::ElementType
ElementKeyword::determine_elementType(const std::string& _keyword_name) const
{
  auto line_lower = to_lower_copy(_keyword_name);

  // hardcoding festival !!!!!!! yeah !!!!!!
  if (line_lower[9] == 's' && line_lower[10] == 'h' && line_lower[11] == 'e' &&
      line_lower[12] == 'l' && line_lower[13] == 'l')
    return Element::ElementType::SHELL;

  else if (line_lower[9] == 'b' && line_lower[10] == 'e' &&
           line_lower[11] == 'a' && line_lower[12] == 'm')
    return Element::ElementType::BEAM;

  else if (line_lower[9] == 's' && line_lower[10] == 'o' &&
           line_lower[11] == 'l' && line_lower[12] == 'i' &&
           line_lower[13] == 'd')
    return Element::ElementType::SOLID;

  else if (line_lower[9] == 't' && line_lower[10] == 's' &&
           line_lower[11] == 'h' && line_lower[12] == 'e' &&
           line_lower[13] == 'l' && line_lower[14] == 'l')
    return Element::ElementType::TSHELL;

  else
    return Element::ElementType::NONE;
}

/** Get all elements of the card
 *
 * @param _type : element type for optional filtering
 * @return elements
 */
std::vector<std::shared_ptr<Element>>
ElementKeyword::get_elements()
{
  std::vector<std::shared_ptr<Element>> elems;
  elems.reserve(elem_indexes_in_card.size());
  for (auto iElement : elem_indexes_in_card)
    elems.push_back(db_elems->get_elementByIndex(type, iElement));

  std::move(elems);
}

/** Convert keyword into a string
 *
 * @return str
 */
std::string
ElementKeyword::str()
{
  // build header
  std::stringstream ss;
  for (const auto& entry : lines)
    ss << entry << '\n';

  // do the thing
  ss.precision(7); // float
  for (size_t iElement = 0; iElement < elem_indexes_in_card.size();
       ++iElement) {

    // this is faster than getting nodes
    auto element = get_elementByIndex(iElement);
    auto elem_nodes = element->get_nodes();

    ss << std::setw(field_size) << element->get_elementID()
       << std::setw(field_size) << elem_part_ids[iElement]
       << std::setw(field_size) << elem_nodes[0]->get_nodeID()
       << std::setw(field_size) << elem_nodes[1]->get_nodeID()
       << unparsed_element_data[iElement] << '\n';
  }

  // trailing lines
  for (const auto& line : trailing_lines)
    ss << line << '\n';

  return ss.str();
}

} // namespace:qd