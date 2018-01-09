
#include <iomanip>

#ifdef QD_USE_ABSL
#include <absl/strings/numbers.h>
#include <absl/strings/string_view.h>
#endif

#include <dyna_cpp/dyna/KeyFile.hpp>
#include <dyna_cpp/dyna/NodeKeyword.hpp>

namespace qd {

NodeKeyword::NodeKeyword(DB_Nodes* _db_nodes,
                         const std::vector<std::string>& _lines,
                         int64_t _iLine)
  : Keyword(_lines, _iLine)
  , db_nodes(_db_nodes)
{

  // prepare extraction
  int32_t node_id;
  std::vector<float> coords(3);
  size_t field_size_nodes = has_long_fields() ? 16 : 8; // WTF why not 10 ?!?!
  auto field_size_nodes_x2 = 2 * field_size_nodes;
  auto field_size_nodes_x3 = 3 * field_size_nodes;
  auto field_size_nodes_x5 = 5 * field_size_nodes;

  // find first card line
  size_t header_size = iCard_to_iLine(0, false);
  size_t iLine = header_size;

  // extract node data
  for (; iLine < lines.size(); ++iLine) {

    const auto& line = lines[iLine];
    auto line_trimmed = trim_copy(line);

    if (line_trimmed.empty() || is_keyword(line) || is_comment(line))
      break;

// parse line
#ifdef QD_USE_ABSL
    bool is_ok = true;
    auto line_view = absl::string_view(line);
    is_ok &= absl::SimpleAtoi(line_view.substr(0, field_size_nodes), &node_id);
    is_ok &= absl::SimpleAtof(
      line_view.substr(field_size_nodes, field_size_nodes_x2), &coords[0]);
    is_ok &= absl::SimpleAtof(
      line_view.substr(field_size_nodes_x3, field_size_nodes_x2), &coords[1]);
    is_ok &= absl::SimpleAtof(
      line_view.substr(field_size_nodes_x5, field_size_nodes_x2), &coords[2]);
    if (is_ok) {
      db_nodes->add_node(node_id, coords);
      node_indexes_in_card.push_back(db_nodes->get_nNodes() - 1);
    } else
      std::cout << "Failed to parse node in line: "
                << (static_cast<size_t>(line_index) + iLine) << '\n';
#else
    try {
      node_id = std::stoi(line.substr(0, field_size_nodes));
      coords[0] = std::stof(line.substr(field_size_nodes, field_size_nodes_x2));
      coords[1] =
        std::stof(line.substr(field_size_nodes_x3, field_size_nodes_x2));
      coords[2] =
        std::stof(line.substr(field_size_nodes_x5, field_size_nodes_x2));
      db_nodes->add_node(node_id, coords);
      node_indexes_in_card.push_back(db_nodes->get_nNodes() - 1);
    } catch (std::exception& err) {
      std::cout << "Parsing error in line: " << (_iLine + iLine + 1) << '\n'
                << "error:" << err.what() << '\n'
                << "line :" << line << '\n';
    }
#endif
  }

  // push the gear in the rear :P
  for (; iLine < lines.size(); ++iLine)
    trailing_lines.push_back(lines[iLine]);

  // remove all lines below keyword
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
  const size_t id_width = has_long_fields() ? 16 : 8;
  const size_t float_width = 2 * id_width;

  ss.precision(7); // float
  for (size_t iNode = 0; iNode < this->get_nNodes(); ++iNode) {

    auto node = this->get_nodeByIndex(iNode);
    auto coords = node->get_coords()[0];
    ss << std::setw(id_width) << node->get_nodeID() << std::setw(float_width)
       << coords[0] << std::setw(float_width) << coords[1]
       << std::setw(float_width) << coords[2] << '\n';
  }

  // trailing lines
  for (const auto& line : trailing_lines)
    ss << line << '\n';

  return ss.str();
}

} // NAMESPACE:qd