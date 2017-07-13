
#ifndef DB_NODES_HPP
#define DB_NODES_HPP

// forward declarations
class Node;
class FEMFile;
class DB_Elements;

// includes
#include "Node.hpp"
#include <dyna_cpp/utility/PythonUtility.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class DB_Nodes
{
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
  std::shared_ptr<Node> add_node(int _id, std::vector<float> _coords);

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

  // Python API
  std::vector<std::shared_ptr<Node>> get_nodeByID(pybind11::list _ids)
  {
    return this->get_nodeByID(qd::py::container_to_vector<int>(
      _ids, "An entry of the list was not a fully fledged integer."));
  }
  std::vector<std::shared_ptr<Node>> get_nodeByID(pybind11::tuple _ids)
  {
    return this->get_nodeByID(qd::py::container_to_vector<int>(
      _ids, "An entry of the list was not a fully fledged integer."));
  }
  std::vector<std::shared_ptr<Node>> get_nodeByIndex(pybind11::list _ids)
  {
    return this->get_nodeByIndex(qd::py::container_to_vector<int>(
      _ids, "An entry of the list was not a fully fledged integer."));
  }
  std::vector<std::shared_ptr<Node>> get_nodeByIndex(pybind11::tuple _ids)
  {
    return this->get_nodeByIndex(qd::py::container_to_vector<int>(
      _ids, "An entry of the list was not a fully fledged integer."));
  }
};

/** Get the node index from it's id
 *
 * @param T _id : node id
 * @return size_t : node index
 */
template<typename T>
T
DB_Nodes::get_id_from_index(size_t _index)
{
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
template<typename T>
size_t
DB_Nodes::get_index_from_id(T _id)
{
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
 * @return std::shared_ptr<Node> node : pointer to the node or nullptr if node
 * is not existing!
 */
template<typename T>
inline std::shared_ptr<Node>
DB_Nodes::get_nodeByID(T _id)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  const auto& it = this->id2index_nodes.find(_id);
  if (it == this->id2index_nodes.end())
    throw(std::invalid_argument("Node with id " + to_string(_id) +
                                " does not exist"));
  return this->nodes[it->second];
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
 * @param int _index : index of the node
 * @return std::shared_ptr<Node> node : pointer to the node or nullptr if node
 * is not existing!
 */
template<typename T>
inline std::shared_ptr<Node>
DB_Nodes::get_nodeByIndex(T _index)
{
  static_assert(std::is_integral<T>::value, "Integer number required.");

  if (_index >= this->nodes.size())
    throw(std::invalid_argument("Node with index " + to_string(_index) +
                                " does not exist in the db."));
  return this->nodes[_index];
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

#endif
