
#ifndef NODEKEYWORD_HPP
#define NODEKEYWORD_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/dyna/Keyword.hpp>

namespace qd {

class NodeKeyword : public Keyword
{
private:
  std::map<size_t, std::string> comments_in_node_block;
  DB_Nodes* db_nodes;
  std::vector<int32_t> node_indexes_in_card;

public:
  NodeKeyword(DB_Nodes* _db_nodes,
              std::vector<std::string> _lines,
              int64_t _iLine = 0);
  template<typename T>
  std::shared_ptr<Node> add_node(T _id, float _x, float _y, float _z);
  template<typename T>
  std::shared_ptr<Node> get_nodeByIndex(T _index);
  std::vector<std::shared_ptr<Node>> get_nodes();
  inline std::vector<int32_t> get_node_indexes();
};

/** Add a node to the card
 *
 * @param _id id of the node, pray that it is unique
 * @param _x
 * @param _y
 * @param _z
 * @return node ptr
 */
template<typename T>
std::shared_ptr<Node>
NodeKeyword::add_node(T _id, float _x, float _y, float _z)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  node_indexes_in_card.append(db_nodes.size());
  return db_nodes->add_node(static_cast<int32_t>(_id), _x, _y, _z);
}

/** Get a node from its index in the card
 *
 * @param _index index of the node in the card
 * @return  ptr to node
 */
template<typename T>
std::shared_ptr<Node>
get_nodeByIndex(T _index)
{
  _index = index_treatment(_index, node_indexes_in_card.size());
  this->db_nodes->get_nodeByIndex(_index);
}

/** Get the indexes of the nodes in the card
 *
 * @return node_indexes_in_card indexes of nodes in node database
 */
inline std::vector<int32_t>
NodeKeyword::get_node_indexes()
{
  return node_indexes_in_card;
}

/** Get all nodes in the card
 *
 * @return res ptr to nodes in the card
 */
std::vector<std::shared_ptr<Node>>
NodeKeyword::get_nodes()
{
  std::vector<std::shared_ptr<Node>> res;
  for (auto index : node_indexes_in_card)
    res.push_back(this->db_nodes->get_nodeByIndex(index));
  return res;
}

} // NAMESPACE:qd

#endif