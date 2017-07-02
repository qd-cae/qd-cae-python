
#ifndef DB_NODES_HPP
#define DB_NODES_HPP

// forward declarations
class Node;
class FEMFile;
class DB_Elements;

// includes
#include <pybind11/stl.h>
#include <dyna_cpp/utility/PythonUtility.hpp>
#include "Node.hpp"
#include "dyna_cpp/utility/TextUtility.hpp"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class DB_Nodes {
  friend FEMFile;

 private:
  FEMFile* femfile;
  std::unordered_map<int, size_t> id2index_nodes;
  std::vector<std::shared_ptr<Node>> nodes;

 public:
  DB_Nodes(FEMFile* _femfile);
  virtual ~DB_Nodes();
  size_t get_nNodes();
  void reserve(const size_t _size);
  FEMFile* get_femfile();
  Node* add_node(int _id, std::vector<float> _coords);

  template <typename T>
  T get_id_from_index(size_t _id);
  template <typename T>
  size_t get_index_from_id(T _index);

  template <typename T>
  Node* get_nodeByID(T _id);
  template <typename T>
  std::vector<Node*> get_nodeByID(const std::vector<T>& _ids);
  template <typename T>
  std::shared_ptr<Node> get_nodeByID_shared(T _id);
  template <typename T>
  std::vector<std::shared_ptr<Node>> get_nodeByID_shared(
      const std::vector<T>& _ids);
  template <typename T>
  Node* get_nodeByIndex(T _index);

  // Python API
  std::shared_ptr<Node> get_nodeByID_py(int _id) {
    return this->get_nodeByID_shared(_id);
  }
  std::vector<std::shared_ptr<Node>> get_nodeByID_py(pybind11::list _ids) {
    return this->get_nodeByID_shared(qd::py::container_to_vector<int>(
        _ids, "An entry of the list was not a fully fledged integer."));
  }
  std::vector<std::shared_ptr<Node>> get_nodeByID_py(pybind11::tuple _ids) {
    return this->get_nodeByID_shared(qd::py::container_to_vector<int>(
        _ids, "An entry of the list was not a fully fledged integer."));
  }
};

/** Get the node index from it's id
 *
 * @param T _id : node id
 * @return size_t : node index
 */
template <typename T>
T DB_Nodes::get_id_from_index(size_t _index) {
  static_assert(std::is_integral<T>::value, "Integer number required.");

  if (_index >= nodes.size())
    throw(std::invalid_argument("Node with index " + to_string(_index) +
                                " does not exist in the db."));
  return _index;
}

/** Get the node id from it's index
 *
 * @param T _id : node id
 * @return size_t _index : node index
 */
template <typename T>
size_t DB_Nodes::get_index_from_id(T _id) {
  static_assert(std::is_integral<T>::value, "Integer number required.");

  const auto& it = this->id2index_nodes.find(_id);
  if (it == this->id2index_nodes.end())
    throw(std::invalid_argument("Node with id " + to_string(_id) +
                                " does not exist in the db."));
  return it->second;
}

/** Get a node from the node ID.
 *
 * @param T _id : id of the node
 * @return Node* node : pointer to the node or nullptr if node is not existing!
 */
template <typename T>
inline Node* DB_Nodes::get_nodeByID(T _id) {
  static_assert(std::is_integral<T>::value, "Integer number required.");

  const auto& it = this->id2index_nodes.find(_id);
  if (it == this->id2index_nodes.end())
    throw(std::invalid_argument("Node with id " + to_string(_id) +
                                " does not exist in the db."));

  return this->nodes[it->second].get();
}

/** Get a list of nodes from a list of ids
 * @param std::vector<T> ids : list of ids
 * @return std::vector<Node*> nodes
 */
template <typename T>
std::vector<Node*> DB_Nodes::get_nodeByID(const std::vector<T>& _ids) {
  std::vector<Node*> ret;
  for (const auto& id : _ids) {
    ret.push_back(this->get_nodeByID(id));
  }
  return std::move(ret);
}

/** Get a node from the node ID.
 *
 * @param T _id : id of the node
 * @return Node* node : pointer to the node or nullptr if node is not existing!
 */
template <typename T>
inline std::shared_ptr<Node> DB_Nodes::get_nodeByID_shared(T _id) {
  static_assert(std::is_integral<T>::value, "Integer number required.");

  const auto& it = this->id2index_nodes.find(_id);
  if (it == this->id2index_nodes.end())
    throw(std::invalid_argument("Node with ID" + to_string(_id) +
                                "does not exist"));
  return this->nodes[it->second];
}

/** Get a list node from an id list
 *
 * @param std::vector<T> _ids : node ids
 * @return std::vector<std::shared_ptr<Node>> nodes
 */
template <typename T>
inline std::vector<std::shared_ptr<Node>> DB_Nodes::get_nodeByID_shared(
    const std::vector<T>& _ids) {
  static_assert(std::is_integral<T>::value, "Integer number required.");

  std::vector<std::shared_ptr<Node>> ret;
  for (const auto& node_id : _ids) {
    ret.push_back(this->get_nodeByID_shared(node_id));
  }
  return std::move(ret);
}

/** Get a node from the node index.
 *
 * @param int _index : index of the node
 * @return Node* node : pointer to the node or nullptr if node is not existing!
 */
template <typename T>
inline Node* DB_Nodes::get_nodeByIndex(T _index) {
  static_assert(std::is_integral<T>::value, "Integer number required.");

  if (_index >= this->nodes.size())
    throw(std::invalid_argument("Node with index " + to_string(_index) +
                                " does not exist in the db."));
  return this->nodes[_index].get();
}

#endif
