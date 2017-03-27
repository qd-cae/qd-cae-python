
#include <string>
#include "DB_Elements.hpp"
#include "DB_Nodes.hpp"
#include "Node.hpp"

using namespace std;

/*
 * Constructor.
 */
DB_Nodes::DB_Nodes(FEMFile* _femfile){

  this->femfile = _femfile;

}


/*
 * Destructor.
 */
DB_Nodes::~DB_Nodes(){

}


/** Add a node to the db by node-ID and it's
 *
 * @param int _nodeID : id of the node
 * @param vector<float> coords : coordinates of the node
 * @return Node* node : pointer to created instance
 *
 * Returns a pointer to the new node.
 */
Node* DB_Nodes::add_node(int _nodeID, vector<float> coords){

  if(coords.size() != 3){
    throw(string("The node coordinate vector must have length 3."));
  }
  if(_nodeID < 0){
    throw(string("Node-ID may not be negative!"));
  }

  // Check if node already is in map
  if(this->id2index.count(_nodeID) != 0)
    throw(string("Trying to insert a node with same id twice:")+to_string(_nodeID));

  // Create and add new node
  //Node* node = new Node(_nodeID,coords,this);
  unique_ptr<Node> node(new Node(_nodeID,coords,this));
  id2index.insert(pair<int,size_t>(_nodeID,this->nodes.size()));
  this->nodes.push_back(std::move(node));

  return this->nodes.back().get();

}


/*
 * Register the element db in the node db.
 */
void DB_Nodes::set_db_elements(DB_Elements* _db_elements){
  if(_db_elements == NULL)
    throw(string("Setting db_elements=NULL in db_nodes is forbidden."));
  this->db_elements = _db_elements;
}

/*
 * Get the owning d3plot of the db.
 */
FEMFile* DB_Nodes::get_femfile(){
   return this->femfile;
}

/*
 * get the element db.
 */
DB_Elements* DB_Nodes::get_db_elements(){
  return this->db_elements;
}


/*
 * Get the number of nodes in the db.
 */
size_t DB_Nodes::size(){
  if(this->id2index.size() != this->nodes.size())
    throw(string("Node database encountered error: id2index.size() != nodes.size()"));
  return this->nodes.size();
}

/** Reserve memory for incoming nodes
 *
 * @param _size size to reserve for new nodes
 */
void DB_Nodes::reserve(const size_t _size){
  this->nodes.reserve(_size);
}
