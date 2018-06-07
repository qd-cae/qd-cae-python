
#include "DB_Nodes.hpp"
#include "DB_Elements.hpp"
#include "FEMFile.hpp"
#include "Node.hpp"

#include <stdexcept>
#include <string>

namespace qd {

/*
 * Constructor.
 */
DB_Nodes::DB_Nodes(FEMFile* _femfile)
  : femfile(_femfile)
{}

/*
 * Destructor.
 */
DB_Nodes::~DB_Nodes()
{
#ifdef QD_DEBUG
  std::cout << "DB_Nodes::~DB_Nodes called." << std::endl;
#endif
}

/** Add a node to the db by node-ID and it's coordinates
 *
 * @param _nodeID : id of the node
 * @param  coords : coordinates of the node
 * @return node : pointer to created instance
 *
 * Returns a pointer to the new node.
 */
std::shared_ptr<Node>
DB_Nodes::add_node(int32_t _nodeID, const std::vector<float>& coords)
{
  if (coords.size() != 3) {
    throw(std::invalid_argument(
      "The node coordinate std::vector must have length 3."));
  }
  if (_nodeID < 0) {
    throw(std::invalid_argument("Node-ID may not be negative!"));
  }

  std::shared_ptr<Node> node = std::make_shared<Node>(_nodeID, coords, this);

#pragma omp critical
  {
    // std::lock_guard<std::mutex> lock(_node_mutex);

    // check if node already is in map
    if (this->id2index_nodes.count(_nodeID) != 0)
      throw(
        std::invalid_argument("Trying to insert a node with same id twice: " +
                              std::to_string(_nodeID)));

    id2index_nodes.insert(
      std::pair<int32_t, size_t>(_nodeID, this->nodes.size()));
    this->nodes.push_back(node);
  }

  return std::move(node);
}

/** Add a node to the db by node-ID and it's coordinates
 *
 * @param _nodeID : id of the node
 * @param  coords : coordinates of the node
 * @return node : pointer to created instance
 *
 * Returns a pointer to the new node.
 */
std::shared_ptr<Node>
DB_Nodes::add_node(int32_t _nodeID, float _x, float _y, float _z)
{
  if (_nodeID < 0) {
    throw(std::invalid_argument("Node-ID may not be negative!"));
  }

  // Create and add new node
  std::shared_ptr<Node> node =
    std::make_shared<Node>(_nodeID, _x, _y, _z, this);

#pragma omp critical
  {
    // std::lock_guard<std::mutex> lock(_node_mutex);

    // Check if node already is in map
    if (this->id2index_nodes.count(_nodeID) != 0)
      throw(
        std::invalid_argument("Trying to insert a node with same id twice: " +
                              std::to_string(_nodeID)));

    id2index_nodes.insert(
      std::pair<int32_t, size_t>(_nodeID, this->nodes.size()));
    this->nodes.push_back(node);
  }

  return std::move(node);
}

/** Add a node while parsing a keyfile
 *
 * @param _id : node_id
 * @param _coords
 * @return node
 *
 * The special case is, if the node is already existing
 * it's data will jsut be corrected. This makes sense,
 * since if an element is missing nodes on the fly,
 * we simply create dummy ones.
 */
std::shared_ptr<Node>
DB_Nodes::add_node_byKeyFile(int32_t _id, float _x, float _y, float _z)
{
  std::lock_guard<std::mutex> lock(_instance_mutex);
  auto it = id2index_nodes.find(_id);

  // correct existing nodes
  if (it != id2index_nodes.end()) {
    auto node = get_nodeByID(_id);
    node->set_coords(_x, _y, _z);
    return node;
  }
  // create missing node
  else {
    auto node = add_node(_id, _x, _y, _z);
    return node;
  }
}

/** Get the owning d3plot of the db.
 *
 */
FEMFile*
DB_Nodes::get_femfile()
{
  return this->femfile;
}

/**Get the number of nodes in the db.
 *
 * @return nNodes
 */
size_t
DB_Nodes::get_nNodes() const
{
#ifdef QD_DEBUG
  if (this->id2index_nodes.size() != this->nodes.size())
    throw(std::runtime_error("Node database encountered error: "
                             "id2index_nodes.size() != nodes.size()"));
#endif

  return this->nodes.size();
}

/** Reserve memory for incoming nodes
 *
 * @param _size size to reserve for new nodes
 */
void
DB_Nodes::reserve(const size_t _size)
{
  this->nodes.reserve(_size);
}

/** Get the nodes of the databse
 *
 * @return nodes : nodes in the database
 */
std::vector<std::shared_ptr<Node>>
DB_Nodes::get_nodes()
{
  return this->nodes;
}

/** Get nodal data as arrays
 *
 * @param name : name of the field
 * @return tensor : data as tensor
 */
Tensor_ptr<float>
DB_Nodes::get_node_coords()
{
  auto tensor = std::make_shared<Tensor<float>>();

  // no data
  if (nodes.size() == 0)
    return tensor;

  // do the thing
  const auto& first_node_data = nodes[0]->get_coords();

  if (first_node_data.size() == 0)
    return tensor;

  const auto nTimesteps = first_node_data.size();
  const auto nDims = first_node_data[0].size();

  tensor->resize({ nodes.size(), nTimesteps, nDims });
  auto& tensor_data = tensor->get_data();

  for (size_t iNode = 0; iNode < nodes.size(); ++iNode) {
    const auto& series = nodes[iNode]->get_coords();
    const auto offset = iNode * nTimesteps * nDims;
    for (size_t iStep = 0; iStep < series.size(); ++iStep) {
      const auto offset2 = offset + iStep * nDims;
      tensor_data[offset2] = series[iStep][0];
      tensor_data[offset2 + 1] = series[iStep][1];
      tensor_data[offset2 + 2] = series[iStep][2];
    }
  }

  return tensor;
}

Tensor_ptr<float>
DB_Nodes::get_node_velocity()
{
  auto tensor = std::make_shared<Tensor<float>>();

  // no data
  if (nodes.size() == 0)
    return tensor;

  // do the thing
  const auto& first_node_data = nodes[0]->get_vel();

  if (first_node_data.size() == 0)
    return tensor;

  const auto nTimesteps = first_node_data.size();
  const auto nDims = first_node_data[0].size();

  tensor->resize({ nodes.size(), nTimesteps, nDims });
  auto& tensor_data = tensor->get_data();

  for (size_t iNode = 0; iNode < nodes.size(); ++iNode) {
    const auto& series = nodes[iNode]->get_vel();
    const auto offset = iNode * nTimesteps * nDims;
    for (size_t iStep = 0; iStep < series.size(); ++iStep) {
      const auto offset2 = offset + iStep * nDims;
      tensor_data[offset2] = series[iStep][0];
      tensor_data[offset2 + 1] = series[iStep][1];
      tensor_data[offset2 + 2] = series[iStep][2];
    }
  }

  return tensor;
}

Tensor_ptr<float>
DB_Nodes::get_node_acceleration()
{
  auto tensor = std::make_shared<Tensor<float>>();

  // no data
  if (nodes.size() == 0)
    return tensor;

  // do the thing
  const auto& first_node_data = nodes[0]->get_accel();

  if (first_node_data.size() == 0)
    return tensor;

  const auto nTimesteps = first_node_data.size();
  const auto nDims = first_node_data[0].size();

  tensor->resize({ nodes.size(), nTimesteps, nDims });
  auto& tensor_data = tensor->get_data();

  for (size_t iNode = 0; iNode < nodes.size(); ++iNode) {
    const auto& series = nodes[iNode]->get_accel();
    const auto offset = iNode * nTimesteps * nDims;
    for (size_t iStep = 0; iStep < series.size(); ++iStep) {
      const auto offset2 = offset + iStep * nDims;
      tensor_data[offset2] = series[iStep][0];
      tensor_data[offset2 + 1] = series[iStep][1];
      tensor_data[offset2 + 2] = series[iStep][2];
      // std::copy(series[iStep].begin(),
      //           series[iStep].end(),
      //           tensor_data.begin() + iNode * nTimesteps * nDims);
    }
  }

  return tensor;
}

Tensor_ptr<int32_t>
DB_Nodes::get_node_ids()
{
  auto tensor = std::make_shared<Tensor<int32_t>>();

  // no data
  if (nodes.size() == 0)
    return tensor;

  tensor->resize({ nodes.size() }); // <- allocate memory
  auto& tensor_data = tensor->get_data();

  for (size_t iNode = 0; iNode < nodes.size(); ++iNode) {
    tensor_data[iNode] = nodes[iNode]->get_nodeID();
  }

  return tensor;
}

} // namespace qd