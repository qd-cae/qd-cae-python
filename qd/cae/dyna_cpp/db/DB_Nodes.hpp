
#ifndef DB_NODES_HPP
#define DB_NODES_HPP

// includes
#include <cstdint>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/math/Tensor.hpp>
#include <dyna_cpp/utility/containers.hpp>

namespace qd {

// forward declarations
class FEMFile;
class DB_Elements;

class DB_Nodes
{
  friend FEMFile;

private:
  std::mutex _instance_mutex;

  FEMFile* femfile;
  std::unordered_map<int32_t, size_t> id2index_nodes;
  std::vector<std::shared_ptr<Node>> nodes;

  std::unordered_map<std::string, size_t> field_name_to_index;
  std::vector<Tensor_ptr<float>> fields;
  Tensor_ptr<int32_t> node_ids;

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

  // array data
  Tensor_ptr<float> get_node_coords();
  Tensor_ptr<float> get_node_velocity();
  Tensor_ptr<float> get_node_acceleration();
  Tensor_ptr<int32_t> get_node_ids();
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

} // namespace qd

#endif
