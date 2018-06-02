
#include <iomanip>

#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/dyna/keyfile/ElementKeyword.hpp>

namespace qd {

/** Constructor of an ElementKeyword
 * @param _db_elems : parent database
 * @param _lines : lines of the keyword
 * @param_iLine : line index of keyword
 */
ElementKeyword::ElementKeyword(DB_Elements* _db_elems,
                               const std::vector<std::string>& _lines,
                               int64_t _iLine)
  : Keyword(_lines, _iLine)
  , db_elems(_db_elems)
  , element_type(Element::ElementType::NONE)
{
  // keyword type
  kw_type = KeywordType::ELEMENT;
  // element type
  element_type = determine_element_type(get_keyword_name());
}

/** Load the data from the string data
 *
 * This function loads the data from the string data.
 * The string data is removed while the data is being parsed.
 */
void
ElementKeyword::load()
{
  if (db_elems == nullptr)
    return;

  // prepare extraction
  field_size = has_long_fields() ? 16 : 8;

  // what element type do we have?
  auto keyword_name = to_lower_copy(get_keyword_name());
  element_type = determine_element_type(keyword_name);
  if (element_type == Element::ElementType::NONE)
    throw(
      std::invalid_argument("Can not find out, what type of element "
                            "(beam,shell,solid,tshell) the keyword contains:"));

  // do the thing
  switch (element_type) {
    case (Element::ElementType::BEAM):
      parse_elem2(keyword_name, lines);
      break;

    case (Element::ElementType::SHELL):
      parse_elem4(keyword_name, lines);
      break;

    case (Element::ElementType::SOLID):
      parse_elem8(keyword_name, lines);
      break;

    case (Element::ElementType::TSHELL):
      parse_elem4th(keyword_name, lines);
      break;

    default:
      break;
  }
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
  size_t header_size = get_line_index_of_next_card(0);
  size_t iLine = header_size;

  if (header_size == lines.size())
    return;

  // how much data to read
  size_t nAdditionalLines = 0;
  if (_keyword_name_lower.find("thickness", 13) != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("section", 13) != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("scalar", 13) != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("scalr", 13) != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("pid", 13) != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("offset", 13) != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("orientation", 13) != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("warpage", 13) != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("elbow", 13) != std::string::npos)
    ++nAdditionalLines;

#ifdef QD_DEBUG
  std::cout << "ElementKeyword\nname: " << _keyword_name_lower
            << "\nnAdditionalLines: " << nAdditionalLines << '\n';
#endif

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

      db_elems->add_elementByNodeID(
        element_type, element_id, part_id, node_ids);
      unparsed_element_data.push_back(remaining_data);
      elem_indexes_in_card.push_back(
        db_elems->get_element_index_from_id(element_type, element_id));
      elem_part_ids.push_back(part_id);

    } catch (const std::out_of_range& err) {
      std::cout << "Parsing error in line: " << (position + iLine + 1) << '\n'
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

/** Parse the string buffer as shell element
 *
 * @param _keyword_name_lower : keyword name in lowercase
 * @param _lines : lines to parse
 */
void
ElementKeyword::parse_elem4(const std::string& _keyword_name_lower,
                            const std::vector<std::string>& _lines)
{

  // find first card line
  size_t header_size = iCard_to_iLine(0, false);
  size_t iLine = header_size;

  // how much data to read
  size_t nAdditionalLines = 0;
  bool skip_eventually_one_more = false;

  if (_keyword_name_lower.find("thickness", 14) != std::string::npos ||
      _keyword_name_lower.find("beta", 14) != std::string::npos ||
      _keyword_name_lower.find("mcid", 14) != std::string::npos) {
    ++nAdditionalLines;
    // +1 again if n5->n8 (field 7-10) present ...
    skip_eventually_one_more = true;
  } else if (_keyword_name_lower.find("offset", 14) != std::string::npos)
    ++nAdditionalLines;
  else if (_keyword_name_lower.find("dof", 14) != std::string::npos)
    ++nAdditionalLines;
  // composite or composite_long ...
  if (_keyword_name_lower.find("composite", 14) != std::string::npos) {
    std::cout
      << "Warning: No support for parsing mesh from *ELEMENT_SHELL_COMPOSITE."
      << '\n';
    return;
  }

#ifdef QD_DEBUG
  std::cout << "ElementKeyword\nname: " << _keyword_name_lower
            << "\nnAdditionalLines: " << nAdditionalLines << '\n';
#endif

  // extract node data
  std::string remaining_data;
  int32_t element_id;
  int32_t part_id;
  std::vector<int32_t> node_ids(4);
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
      node_ids[2] = std::stoi(line.substr(4 * field_size, field_size));
      node_ids[3] = std::stoi(line.substr(5 * field_size, field_size));

      // save remaining element data
      if (6 * field_size < line.size())
        remaining_data = std::string(line.begin() + 6 * field_size, line.end());
      else
        remaining_data = std::string();

      for (size_t iExtraLine = 0;
           iExtraLine < nAdditionalLines +
                          (skip_eventually_one_more && !remaining_data.empty());
           ++iExtraLine)
        remaining_data += '\n' + lines[iLine + 1 + iExtraLine];

      // do the thing
      db_elems->add_elementByNodeID(
        element_type, element_id, part_id, node_ids);
      unparsed_element_data.push_back(remaining_data);
      elem_indexes_in_card.push_back(
        db_elems->get_element_index_from_id(element_type, element_id));
      elem_part_ids.push_back(part_id);

    } catch (const std::exception& err) {
      std::cout << "Parsing error in line: " << (position + iLine + 1) << '\n'
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

/** Parse the string buffer as solid elements
 *
 * @param _keyword_name_lower : keyword name in lower letters
 * @param _lines : line buffer
 */
void
ElementKeyword::parse_elem8(const std::string& _keyword_name_lower,
                            const std::vector<std::string>& _lines)
{

  // find first card line
  size_t header_size = get_line_index_of_next_card(0);
  size_t iLine = header_size;

  if (header_size == lines.size())
    return;

  // how much data to read
  size_t nAdditionalLines = 0;
  if (_keyword_name_lower.find("t15", 13) != std::string::npos ||
      _keyword_name_lower.find("t20", 13) != std::string::npos ||
      _keyword_name_lower.find("h20", 13) != std::string::npos)
    ++nAdditionalLines;
  else if (_keyword_name_lower.find("p21", 13) != std::string::npos ||
           _keyword_name_lower.find("h27", 13) != std::string::npos)
    nAdditionalLines += 2;
  else if (_keyword_name_lower.find("t40", 13) != std::string::npos)
    nAdditionalLines += 3;
  else if (_keyword_name_lower.find("h64", 13) != std::string::npos)
    nAdditionalLines += 6;
  else if (_keyword_name_lower.find("ortho", 13) != std::string::npos)
    nAdditionalLines += 2;
  else if (_keyword_name_lower.find("dof", 13) != std::string::npos)
    ++nAdditionalLines;

#ifdef QD_DEBUG
  std::cout << "ElementKeyword\nname: " << _keyword_name_lower
            << "\nnAdditionalLines: " << nAdditionalLines << '\n';
#endif

  // extract node data
  int32_t element_id;
  int32_t part_id;
  std::vector<int32_t> node_ids(8);
  std::string remaining_data;
  for (; iLine < lines.size(); iLine += 1 + nAdditionalLines) {

    const auto& line = lines[iLine];
    auto line_trimmed = trim_copy(line);

    if (line_trimmed.empty() || is_keyword(line) || is_comment(line))
      break;

    // parse line
    try {
      element_id = std::stoi(line.substr(0, field_size));
      part_id = std::stoi(line.substr(field_size, field_size));
      if (line_trimmed.size() > 3 * field_size) {
        node_ids[0] = std::stoi(line.substr(2 * field_size, field_size));
        node_ids[1] = std::stoi(line.substr(3 * field_size, field_size));
        node_ids[2] = std::stoi(line.substr(4 * field_size, field_size));
        node_ids[3] = std::stoi(line.substr(5 * field_size, field_size));
        node_ids[4] = std::stoi(line.substr(6 * field_size, field_size));
        node_ids[5] = std::stoi(line.substr(7 * field_size, field_size));
        node_ids[6] = std::stoi(line.substr(8 * field_size, field_size));
        node_ids[7] = std::stoi(line.substr(9 * field_size, field_size));
      } else {
        const auto& next_line = lines[++iLine];
        node_ids[0] = std::stoi(next_line.substr(0, field_size));
        node_ids[1] = std::stoi(next_line.substr(1 * field_size, field_size));
        node_ids[2] = std::stoi(next_line.substr(2 * field_size, field_size));
        node_ids[3] = std::stoi(next_line.substr(3 * field_size, field_size));
        node_ids[4] = std::stoi(next_line.substr(4 * field_size, field_size));
        node_ids[5] = std::stoi(next_line.substr(5 * field_size, field_size));
        node_ids[6] = std::stoi(next_line.substr(6 * field_size, field_size));
        node_ids[7] = std::stoi(next_line.substr(7 * field_size, field_size));
        if (next_line.size() > 8 * field_size)
          remaining_data =
            std::string(next_line.begin() + 8 * field_size, next_line.end());
      }

      // save remaining element data
      for (size_t iExtraLine = 0; iExtraLine < nAdditionalLines; ++iExtraLine)
        remaining_data += '\n' + lines[iLine + 1 + iExtraLine];

      db_elems->add_elementByNodeID(
        element_type, element_id, part_id, node_ids);
      unparsed_element_data.push_back(remaining_data);
      elem_indexes_in_card.push_back(
        db_elems->get_element_index_from_id(element_type, element_id));
      elem_part_ids.push_back(part_id);

    } catch (const std::out_of_range& err) {
      std::cout << "Parsing error in line: " << (position + iLine + 1) << '\n'
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

/** Parse the string buffer as thick elements
 *
 * @param _keyword_name_lower : keyword name in lower letters
 * @param _lines : line buffer
 */
void
ElementKeyword::parse_elem4th(const std::string& _keyword_name_lower,
                              const std::vector<std::string>& _lines)
{

  // find first card line
  size_t header_size = get_line_index_of_next_card(0);
  size_t iLine = header_size;

  if (header_size == lines.size())
    return;

  // how much data to read
  size_t nAdditionalLines = 0;
  if (_keyword_name_lower.find("beta", 14) != std::string::npos)
    ++nAdditionalLines;
  if (_keyword_name_lower.find("composite", 14) != std::string::npos) {
    std::cout
      << "Warning: No support for parsing mesh from *ELEMENT_TSHELL_COMPOSITE."
      << '\n';
    return;
  }

  // extract node data
  int32_t element_id;
  int32_t part_id;
  std::vector<int32_t> node_ids(8);
  std::string remaining_data;
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
      node_ids[2] = std::stoi(line.substr(4 * field_size, field_size));
      node_ids[3] = std::stoi(line.substr(5 * field_size, field_size));
      node_ids[4] = std::stoi(line.substr(6 * field_size, field_size));
      node_ids[5] = std::stoi(line.substr(7 * field_size, field_size));
      node_ids[6] = std::stoi(line.substr(8 * field_size, field_size));
      node_ids[7] = std::stoi(line.substr(9 * field_size, field_size));

      // save remaining element data
      for (size_t iExtraLine = 0; iExtraLine < nAdditionalLines; ++iExtraLine)
        remaining_data += '\n' + lines[iLine + 1 + iExtraLine];

      db_elems->add_elementByNodeID(
        element_type, element_id, part_id, node_ids);
      unparsed_element_data.push_back(remaining_data);
      elem_indexes_in_card.push_back(
        db_elems->get_element_index_from_id(element_type, element_id));
      elem_part_ids.push_back(part_id);

    } catch (const std::out_of_range& err) {
      std::cout << "Parsing error in line: " << (position + iLine + 1) << '\n'
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
ElementKeyword::determine_element_type(const std::string& _keyword_name) const
{
  if (_keyword_name.size() <= 9)
    return Element::ElementType::NONE;

  auto line_lower = to_lower_copy(_keyword_name);

  // hardcoding festival !!!!!!! yeah !!!!!!
  if (line_lower.compare(9, 5, "shell") == 0)
    return Element::ElementType::SHELL;

  else if (line_lower.compare(9, 4, "beam") == 0)
    return Element::ElementType::BEAM;

  else if (line_lower.compare(9, 5, "solid") == 0)
    return Element::ElementType::SOLID;

  else if (line_lower.compare(9, 6, "tshell") == 0)
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
    elems.push_back(db_elems->get_elementByIndex(element_type, iElement));

  return std::move(elems);
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
  if (element_type == Element::ElementType::BEAM)
    keyword_elem2_str(ss);
  else if (element_type == Element::ElementType::SHELL)
    keyword_elem4_str(ss);
  else if (element_type == Element::ElementType::SOLID)
    keyword_elem8_str(ss);

  // trailing lines
  for (const auto& line : trailing_lines)
    ss << line << '\n';

  return ss.str();
}

/** Convert the keyword to string for elem2
 * @param _ss :  stringstream to put stuff in
 */
void
ElementKeyword::keyword_elem2_str(std::stringstream& _ss)
{

  _ss.precision(7); // float
  for (size_t iElement = 0; iElement < elem_indexes_in_card.size();
       ++iElement) {

    // this is faster than getting nodes
    auto element = get_elementByIndex(iElement);
    auto elem_nodes = element->get_nodes();

    _ss << std::setw(field_size) << element->get_elementID()
        << std::setw(field_size) << elem_part_ids[iElement]
        << std::setw(field_size) << elem_nodes[0]->get_nodeID()
        << std::setw(field_size) << elem_nodes[1]->get_nodeID()
        << unparsed_element_data[iElement] << '\n';
  }
}

/** Convert the keyword to string for elem2
 * @param _ss :  stringstream to put stuff in
 */
void
ElementKeyword::keyword_elem4_str(std::stringstream& _ss)
{
  // index for comparison, whether we have 3 or 4 nodes
  // if we have 3 then the last is printed twice
  constexpr size_t magic_index = 2;

  _ss.precision(7); // float
  for (size_t iElement = 0; iElement < elem_indexes_in_card.size();
       ++iElement) {

    // element and nodes
    auto element = get_elementByIndex(iElement);
    auto elem_nodes = element->get_nodes();

    // write stuff
    _ss
      << std::setw(field_size) << element->get_elementID()
      << std::setw(field_size) << elem_part_ids[iElement]
      << std::setw(field_size) << elem_nodes[0]->get_nodeID()
      << std::setw(field_size) << elem_nodes[1]->get_nodeID()
      << std::setw(field_size) << elem_nodes[2]->get_nodeID()
      << std::setw(field_size)
      << elem_nodes[std::max(elem_nodes.size() - 1, magic_index)]->get_nodeID();

    if (unparsed_element_data.size() != 0)
      _ss << unparsed_element_data[iElement] << '\n';
  }
}

/** Convert the keyword to string for elem4th
 *
 * @param _ss :  stringstream to put stuff in
 */
void
ElementKeyword::keyword_elem4th_str(std::stringstream& _ss)
{

  _ss.precision(7); // float
  for (size_t iElement = 0; iElement < elem_indexes_in_card.size();
       ++iElement) {

    // element and nodes
    auto element = get_elementByIndex(iElement);
    auto elem_nodes = element->get_nodes();

    _ss << std::setw(field_size) << element->get_elementID()
        << std::setw(field_size) << elem_part_ids[iElement];

    if (elem_nodes.size() == 6) {
      _ss << std::setw(field_size) << elem_nodes[0]->get_nodeID()
          << std::setw(field_size) << elem_nodes[1]->get_nodeID()
          << std::setw(field_size) << elem_nodes[2]->get_nodeID()
          << std::setw(field_size) << elem_nodes[2]->get_nodeID()
          << std::setw(field_size) << elem_nodes[3]->get_nodeID()
          << std::setw(field_size) << elem_nodes[4]->get_nodeID()
          << std::setw(field_size) << elem_nodes[5]->get_nodeID()
          << std::setw(field_size) << elem_nodes[5]->get_nodeID();
    } else if (elem_nodes.size() == 8) {
      _ss << std::setw(field_size) << elem_nodes[0]->get_nodeID()
          << std::setw(field_size) << elem_nodes[1]->get_nodeID()
          << std::setw(field_size) << elem_nodes[2]->get_nodeID()
          << std::setw(field_size) << elem_nodes[3]->get_nodeID()
          << std::setw(field_size) << elem_nodes[4]->get_nodeID()
          << std::setw(field_size) << elem_nodes[5]->get_nodeID()
          << std::setw(field_size) << elem_nodes[6]->get_nodeID()
          << std::setw(field_size) << elem_nodes[7]->get_nodeID();
    } else {
      std::cout << "Warning: thick shell with id " << element->get_elementID()
                << " has an invalid number of nodes: " << elem_nodes.size()
                << '\n';
    }

    // write stuff
    if (unparsed_element_data.size() != 0)
      _ss << unparsed_element_data[iElement] << '\n';
  }
}

/** Convert the keyword to string for elem2
 * @param _ss :  stringstream to put stuff in
 */
void
ElementKeyword::keyword_elem8_str(std::stringstream& _ss)
{

  _ss.precision(7); // float
  for (size_t iElement = 0; iElement < elem_indexes_in_card.size();
       ++iElement) {

    // element and nodes
    auto element = get_elementByIndex(iElement);
    auto elem_nodes = element->get_nodes();

    _ss << std::setw(field_size) << element->get_elementID()
        << std::setw(field_size) << elem_part_ids[iElement];

    for (auto& node : elem_nodes)
      _ss << std::setw(field_size) << node->get_nodeID();

    for (size_t iNode = 0; iNode < 8 - elem_nodes.size(); ++iNode) {
      _ss << std::setw(field_size) << elem_nodes.back()->get_nodeID();
    }

    // write stuff
    if (unparsed_element_data.size() != 0)
      _ss << unparsed_element_data[iElement] << '\n';
  }
}

} // namespace:qd