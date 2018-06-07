
#ifndef NODEKEYWORD_HPP
#define NODEKEYWORD_HPP

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/dyna/keyfile/Keyword.hpp>

namespace qd {

class NodeKeyword : public Keyword
{
private:
  DB_Nodes* db_nodes;
  std::vector<int32_t> node_ids_in_card;
  std::vector<std::string> unparsed_node_data;
  std::vector<std::string> trailing_lines;

  std::mutex _instance_mutex;

public:
  explicit NodeKeyword(DB_Nodes* _db_nodes,
                       const std::vector<std::string>& _lines,
                       int64_t _iLine = 0);
  void load();
  template<typename T>
  std::shared_ptr<Node> add_node(T _id, float _x, float _y, float _z);
  template<typename T>
  std::shared_ptr<Node> add_node(T _id,
                                 float _x,
                                 float _y,
                                 float _z,
                                 const std::string& _additional_card_data = "");

  template<typename T>
  std::shared_ptr<Node> get_nodeByIndex(T _index);
  inline std::vector<std::shared_ptr<Node>> get_nodes();
  inline const std::vector<int32_t>& get_node_ids();
  inline size_t get_nNodes() const;
  std::string str() override;

  inline std::vector<std::string> get_failed_lines();
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

  auto node = db_nodes->add_node(static_cast<int32_t>(_id), _x, _y, _z);
  node_ids_in_card.push_back(node->get_nodeID());
  unparsed_node_data.push_back("");
  return node;
}

/** Add a node with additional card data
 *
 * @param _id id of the node, pray that it is unique
 * @param _x
 * @param _y
 * @param _z
 * @param _additional_card_data : string appended to the output
 * @return node ptr
 *
 */
template<typename T>
std::shared_ptr<Node>
NodeKeyword::add_node(T _id,
                      float _x,
                      float _y,
                      float _z,
                      const std::string& _additional_card_data)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  auto node = db_nodes->add_node(static_cast<int32_t>(_id), _x, _y, _z);
  node_ids_in_card.push_back(node->get_nodeID());
  unparsed_node_data.push_back(_additional_card_data);
  return node;
}

/** Get a node from its index in the card
 *
 * @param _index index of the node in the card
 * @return  ptr to node
 */
template<typename T>
std::shared_ptr<Node>
NodeKeyword::get_nodeByIndex(T _index)
{
  _index = index_treatment(_index, node_ids_in_card.size());

  if (_index > node_ids_in_card.size())
    throw(std::invalid_argument(
      "The card has only " + std::to_string(get_nNodes()) +
      " nodes, you asked for number " + std::to_string(_index)));

  return db_nodes->get_nodeByID(static_cast<int32_t>(node_ids_in_card[_index]));
}

/** Get the indexes of the nodes of this card in the database
 *
 * @return node_indexes_in_card indexes of nodes in node database
 */
inline const std::vector<int32_t>&
NodeKeyword::get_node_ids()
{
  return node_ids_in_card;
}

/** Get all nodes in the card
 *
 * @return res ptr to nodes in the card
 */
std::vector<std::shared_ptr<Node>>
NodeKeyword::get_nodes()
{
  std::vector<std::shared_ptr<Node>> res;
  res.reserve(node_ids_in_card.size());
  for (auto id : node_ids_in_card)
    res.push_back(this->db_nodes->get_nodeByID(id));
  return std::move(res);
}

/** Get the number of nodes in the card
 * @return nNodes
 */
size_t
NodeKeyword::get_nNodes() const
{
  return this->node_ids_in_card.size();
}

} // NAMESPACE:qd

#endif