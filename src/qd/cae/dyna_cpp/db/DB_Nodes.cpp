
#include <string>
#include "DB_Elements.hpp"
#include "DB_Nodes.hpp"
#include "Node.hpp"
#include "../utility/TextUtility.hpp"

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

  // Delete Nodes
  for (std::map<int,Node*>::iterator it=nodesByID.begin(); it!=nodesByID.end(); ++it){
    delete it->second;
  }

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
  if(nodesByID.count(_nodeID) != 0)
    throw(string("Trying to insert a node with same id twice:")+to_string(_nodeID));

  // Create and add new node
  Node* node = new Node(_nodeID,coords,this);
  this->nodesByID.insert(pair<int,Node*>(_nodeID,node));
  indexes2ids.push_back(_nodeID);

  return node;

}


/** Get a node from the node ID.
 *
 * @param int _nodeID : id of the node
 * @return Node* node : pointer to the node or NULL if node is not existing!
 */
Node* DB_Nodes::get_nodeByID(int nodeID){

  map<int,Node*>::iterator it = this->nodesByID.find(nodeID);
  if(it == nodesByID.end())
    return NULL;
  return it->second;

}


/** Get a node from the node index.
 *
 * @param int _nodeIndex : index of the node
 * @return Node* node : pointer to the node or NULL if node is not existing!
 */
Node* DB_Nodes::get_nodeByIndex(int _nodeIndex){

  map<int,Node*>::iterator it =this->nodesByID.find(indexes2ids[_nodeIndex]);
  if(it == nodesByID.end())
     return NULL;
  return it->second;

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
  if(indexes2ids.size() != nodesByID.size())
    throw(string("Node database encountered error: indexes2ids.size() != nodesByID.size()"));
  return indexes2ids.size();
}
