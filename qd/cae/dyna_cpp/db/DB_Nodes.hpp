
#ifndef DB_NODES_HPP
#define DB_NODES_HPP

// includes
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/utility/containers.hpp>

namespace qd {

// forward declarations
class FEMFile;
class DB_Elements;

class DB_Nodes
{
  friend FEMFile;

private:
  FEMFile* femfile;
  std::unordered_map<int32_t, size_t> id2index_nodes;
  std::vector<std::shared_ptr<Node>> nodes;

public:
  explicit DB_Nodes(FEMFile* _femfile);
  virtual ~DB_Nodes();
  size_t get_nNodes() const;
  void reserve(const size_t _size);
  FEMFile* get_femfile();
  std::shared_ptr<Node> add_node(int32_t _id,
                                 const std::vector<float>& _coords);
  std::shared_ptr<Node> add_node(int32_t _id, float _x, float _y, float _z);
  std::shared_ptr<Node> add_node_byKeyFile(int32_t _id,
                                           float _x,
                                           float _y,
                                           float _z);

  template<typename T>
  T get_id_from_index(size_t _id);
  template<typename T>
  size_t get_index_from_id(T _index);

  std::vector<std::shared_ptr<Node>> get_nodes();
  template<typename T>
  std::shared_ptr<Node> get_nodeByID(T _id);
  template<typename T>
  std::vector<std::shared_ptr<Node>> get_nodeByID(const std::vector<T>& _ids);
  template<typename T>
  std::shared_ptr<Node> get_nodeByIndex(T _index);
  template<typename T>
  std::vector<std::shared_ptr<Node>> get_nodeByIndex(
    const std::vector<T>& _ids);
  template<typename T>
  std::shared_ptr<Node> get_nodeByIndex_nothrow(T _index);

  template<typename T>
  void remove_nodeByID(std::vector<T> _indexes);
};

/** Get the node index from it's id
 *
 * @param T _id : node id
 * @return size_t : node index
 */
template<typename T>
inline T
DB_Nodes::get_id_from_index(size_t _index)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  return this->get_nodeByIndex(_index)->get_nodeID();
}

/** Get the node id from it's index
 *
 * @param T _id : node id
 * @return size_t _index : node index
 */
template<typename T>
inline size_t
DB_Nodes::get_index_from_id(T _id)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  const auto& it = this->id2index_nodes.find(_id);
  if (it == this->id2index_nodes.end())
    throw(std::invalid_argument("Could not find node with id " +
                                std::to_string(_id)));

  return it->second;
}

/** Get a node from the node ID.
 *
 * @param T _id : id of the node
 * @return std::shared_ptr<Node> node : pointer to the node or nullptr if node
 * is not existing!
 */
template<typename T>
inline std::shared_ptr<Node>
DB_Nodes::get_nodeByID(T _id)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  return this->get_nodeByIndex(this->get_index_from_id(_id));
}

/** Get a list node from an id list
 *
 * @param std::vector<T> _ids : node ids
 * @return std::vector<std::shared_ptr<Node>> nodes
 */
template<typename T>
inline std::vector<std::shared_ptr<Node>>
DB_Nodes::get_nodeByID(const std::vector<T>& _ids)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  std::vector<std::shared_ptr<Node>> ret;
  for (const auto& node_id : _ids) {
    ret.push_back(this->get_nodeByID(node_id));
  }

  return std::move(ret);
}

/** Get a node from the node index.
 *
 * @param int32_t _index : index of the node
 * @return std::shared_ptr<Node> node : pointer to the node or nullptr if node
 * is not existing!
 */
template<typename T>
inline std::shared_ptr<Node>
DB_Nodes::get_nodeByIndex(T _index)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  if (_index >= 0 && static_cast<size_t>(_index) < nodes.size())
    return nodes[_index];
  else
    throw(std::invalid_argument("Could not find node with index " +
                                std::to_string(_index)));
}

/** Get a node from the node index.
 *
 * @param int32_t _index : index of the node
 * @return std::shared_ptr<Node> node : pointer to the node or nullptr if node
 * is not existing!
 */
template<typename T>
inline std::shared_ptr<Node>
DB_Nodes::get_nodeByIndex_nothrow(T _index)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  if (_index >= 0 && _index < nodes.size())
    return nodes[_index];
  else
    return nullptr;
}

/** Get a list of node from an index list
 *
 * @param std::vector<T> _indexes : node indexes
 * @return std::vector<std::shared_ptr<Node>> nodes
 */
template<typename T>
inline std::vector<std::shared_ptr<Node>>
DB_Nodes::get_nodeByIndex(const std::vector<T>& _indexes)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  std::vector<std::shared_ptr<Node>> ret;
  for (const auto index : _indexes) {
    ret.push_back(this->get_nodeByIndex(index));
  }

  return std::move(ret);
}

/** Remove elements by their indexes
 *
 * @param _indexes
 */
template<typename T>
void
DB_Nodes::remove_nodeByID(std::vector<T> _node_ids)
{

  if (_node_ids.empty())
    return;

  auto db_elements = get_femfile()->get_db_elements();

  // make indexes unique and sort
  std::sort(_node_ids.begin(), _node_ids.end());
  _node_ids.erase(std::unique(_node_ids.begin(), _node_ids.end()),
                  _node_ids.end());

  // find elements belonging to nodes
  std::unordered_set<std::shared_ptr<Element>> elems_to_delete;
  for (auto node_id : _node_ids) {
    auto node = get_nodeByID(node_id);

    for (auto node_elem : node->get_elements())
      elems_to_delete.insert(node_elem);
  }

  // delete elements
  db_elements->delete_elements(elemes_to_delete);

  // delete nodes
  for (auto node_id : _node_ids)
    id2index_nodes.erase(node->get_nodeID());
  vector_remove_indexes(nodes, _indexes, false);
}

} // namespace qd

#endif
