
#include <absl/strings/numbers.h>
#include <absl/strings/string_view.h>
#include <dyna_cpp/dyna/NodeKeyword.hpp>

namespace qd {

NodeKeyword::NodeKeyword(DB_Nodes* _db_nodes,
                         std::vector<std::string> _lines,
                         int64_t _iLine)
  : Keyword(_lines, _iLine)
  , db_nodes(_db_nodes)
{

  // find beginning of data
  size_t iLine;
  for (iLine = 0; iLine < lines.size(); ++iLine)
    if (!is_comment(lines[iLine]) && !is_keyword(lines[iLine]))
      break;

  // prepare extraction
  int32_t node_id;
  std::vector<float> coords(3);
  auto field_size_x2 = 2 * field_size;
  auto field_size_x3 = 3 * field_size;
  auto field_size_x5 = 5 * field_size;

  // extract node data
  for (size_t iLine = lines.size() - 1; iLine > 0; --iLine) {

    const auto& line = lines[iLine];

    if (line.empty())
      continue;

    if (is_keyword(line))
      break;

    // save comments inbetween
    if (is_comment(line)) {
      comments_in_node_block.insert(std::make_pair(iLine, line));
      continue;
    }

    // parse line
    bool is_ok = true;
    auto line_view = absl::string_view(line);
    is_ok &= absl::SimpleAtoi(line_view.substr(0, field_size), &node_id);
    is_ok &=
      absl::SimpleAtof(line_view.substr(field_size, field_size_x2), &coords[0]);
    is_ok &= absl::SimpleAtof(line_view.substr(field_size_x3, field_size_x2),
                              &coords[1]);
    is_ok &= absl::SimpleAtof(line_view.substr(field_size_x5, field_size_x2),
                              &coords[2]);

    /*
    node_id = std::stoi(line.substr(0, field_size));
    coords[0] = std::stof(line.substr(field_size, field_size_x2));
    coords[1] = std::stof(line.substr(field_size_x3, field_size_x2));
    coords[2] = std::stof(line.substr(field_size_x5, field_size_x2));
    */

    // add node
    if (is_ok)
      db_nodes->add_node(node_id, coords);
    else
      std::cout << "Failed to parse node in line: "
                << (static_cast<size_t>(line_index) + iLine) << '\n';
  }

  // remove all lines below keyword
  lines.resize(iLine + 1);
}

} // NAMESPACE:qd