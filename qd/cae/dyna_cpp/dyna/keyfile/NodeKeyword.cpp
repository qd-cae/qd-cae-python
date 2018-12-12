
#include <iomanip>

#include <dyna_cpp/dyna/keyfile/KeyFile.hpp>
#include <dyna_cpp/dyna/keyfile/NodeKeyword.hpp>

namespace qd {

/** Constructor of a NodeKeyword
 *
 * @param _db_nodes : parent database
 * @param _lines : line buffer
 * @param _iLine : line (sorting) index
 */
NodeKeyword::NodeKeyword(DB_Nodes* _db_nodes,
                         const std::vector<std::string>& _lines,
                         int64_t _iLine)
  : Keyword(_lines, _iLine)
  , db_nodes(_db_nodes)
{
  field_size = 8;
  kw_type = KeywordType::NODE;
}

/** Load the data from the string data
 *
 * @param _db_nodes : database to put nodes in
 *
 * This function loads the data from the string data.
 * The string data is removed while the data is being parsed.
 */
void
NodeKeyword::load()
{
  std::lock_guard<std::mutex> lock(_instance_mutex);

  if (db_nodes == nullptr)
    return;

  // find first card line
  size_t header_size = get_line_index_of_next_card(0);
  size_t iLine = header_size;

  if (iLine == lines.size())
    return;

  // extra card treatment
  auto keyword_name_lower = to_lower_copy(get_keyword_name());

  // extract node data
  std::string remaining_data;
  int32_t node_id;
  std::vector<float> coords(3);
  field_size = has_long_fields() ? 16 : 8;
  auto field_size_x2 = 2 * field_size;
  auto field_size_x3 = 3 * field_size;
  auto field_size_x5 = 5 * field_size;
  auto field_size_x7 = 7 * field_size;

  for (; iLine < lines.size(); ++iLine) {

    const auto& line = lines[iLine];
    auto line_trimmed = trim_copy(line);

    if (line_trimmed.empty() || is_keyword(line) || is_comment(line))
      break;

    // parse line
    try {

      // node stuff
      node_id = std::stoi(line.substr(0, field_size));
      // wtf optional coordinates ?!?!?! should not cause too many cache misses
      if (field_size < line.size())
        coords[0] = std::stof(line.substr(field_size, field_size_x2));
      else
        coords[0] = 0.f;
      if (field_size_x3 < line.size())
        coords[1] = std::stof(line.substr(field_size_x3, field_size_x2));
      else
        coords[1] = 0.f;
      if (field_size_x5 < line.size())
        coords[2] = std::stof(line.substr(field_size_x5, field_size_x2));
      else
        coords[2] = 0.f;

      // remainder
      if (field_size_x7 < line.size())
        remaining_data = std::string(line.begin() + field_size_x7, line.end());
      else
        remaining_data = std::string();

      db_nodes->add_node(node_id, coords);
      node_ids_in_card.push_back(node_id);
      unparsed_node_data.push_back(remaining_data);

    } catch (std::exception& err) {
      std::cout << "Parsing error in line: "
                << (static_cast<size_t>(position) + 1 + iLine) << '\n'
                << "error:" << err.what() << '\n'
                << "line :" << line << '\n';
    }
  }

  // push the gear in the rear :P
  for (; iLine < lines.size(); ++iLine)
    trailing_lines.push_back(lines[iLine]);

  // remove all lines below keyword (except for failed ones)
  lines.resize(header_size);
}

/** Get the keyword as a string
 *
 * @return keyword as string
 */
std::string
NodeKeyword::str()
{
  // build header
  std::stringstream ss;
  for (const auto& entry : lines)
    ss << entry << '\n';

  // insert nodes
  const size_t id_width = field_size;
  const size_t float_width = 2 * field_size;

  ss.precision(7); // float
  for (size_t iNode = 0; iNode < this->get_nNodes(); ++iNode) {

    auto node = this->get_nodeByIndex(iNode);
    auto coords = node->get_coords()[0];
    ss << std::setw(id_width) << node->get_nodeID() << std::setw(float_width)
       << coords[0] << std::setw(float_width) << coords[1]
       << std::setw(float_width) << coords[2] << unparsed_node_data[iNode]
       << '\n';
  }

  // trailing lines
  for (const auto& line : trailing_lines)
    ss << line << '\n';

  return ss.str();
}

} // NAMESPACE:qd