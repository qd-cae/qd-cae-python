
#include "DB_Nodes.hpp"
#include "DB_Elements.hpp"
#include "FEMFile.hpp"
#include "Node.hpp"

#include <stdexcept>
#include <string>

using namespace std;

/*
 * Constructor.
 */
DB_Nodes::DB_Nodes(FEMFile* _femfile) : femfile(_femfile) {}

/*
 * Destructor.
 */
DB_Nodes::~DB_Nodes() {
#ifdef QD_DEBUG
  std::cout << "DB_Nodes::~DB_Nodes called." << std::endl;
#endif
}

/** Add a node to the db by node-ID and it's
 *
 * @param int _nodeID : id of the node
 * @param vector<float> coords : coordinates of the node
 * @return Node* node : pointer to created instance
 *
 * Returns a pointer to the new node.
 */
Node* DB_Nodes::add_node(int _nodeID, vector<float> coords) {
  if (coords.size() != 3) {
    throw(std::invalid_argument(
        "The node coordinate vector must have length 3."));
  }
  if (_nodeID < 0) {
    throw(std::invalid_argument("Node-ID may not be negative!"));
  }

  // Check if node already is in map
  if (this->id2index_nodes.count(_nodeID) != 0)
    throw(std::invalid_argument("Trying to insert a node with same id twice:" +
                                to_string(_nodeID)));

  // Create and add new node
  shared_ptr<Node> node = std::make_shared<Node>(_nodeID, coords, this);

  id2index_nodes.insert(pair<int, size_t>(_nodeID, this->nodes.size()));
  this->nodes.push_back(std::move(node));

  return this->nodes.back().get();
}

/*
 * Get the owning d3plot of the db.
 */
FEMFile* DB_Nodes::get_femfile() { return this->femfile; }

/*
 * Get the number of nodes in the db.
 */
size_t DB_Nodes::get_nNodes() {
  if (this->id2index_nodes.size() != this->nodes.size())
    throw(
        std::runtime_error("Node database encountered error: "
                           "id2index_nodes.size() != nodes.size()"));
  return this->nodes.size();
}

/** Reserve memory for incoming nodes
 *
 * @param _size size to reserve for new nodes
 */
void DB_Nodes::reserve(const size_t _size) { this->nodes.reserve(_size); }
