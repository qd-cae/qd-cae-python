
#ifndef NODEKEYWORD_HPP
#define NODEKEYWORD_HPP

#include <cstdint>

#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/dyna/Keyword.hpp>

namespace qd {

class NodeKeyword : public Keyword
{
private:
  std::vector<std::string> comments_in_node_block;
  std::shared_ptr<DB_Nodes> db_nodes;
  std::vector<int32_t> node_indexes_in_card;

public:
  NodeKeyword(std::shared_ptr<DB_Nodes> _db_nodes,
              std::vector<std::string> _lines,
              int64_t _iLine = 0);
  template<typename T>
  std::shared_ptr<Node> get_nodeByIndex(T _index);
};

template<typename T>
std::shared_ptr<Node>
get_nodeByIndex(T _index)
{
  _index = index_treatment(_index, node_indexes_in_card.size());

  this->db_nodes->get_nodeByIndex(_index);
}

} // NAMESPACE:qd

#endif