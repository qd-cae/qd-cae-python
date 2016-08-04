
#include <string>
#include "DB_Elements.h"
#include "DB_Nodes.h"
#include "../dyna/d3plot.h"
#include "Node.h"
#include "../utility/TextUtility.h"

/*
 * Constructor.
 */
DB_Nodes::DB_Nodes(D3plot* d3plot){

  this->nodesByIndex = map<int,Node*>();
  this->nodesByID = map<int,Node*>();
  this->d3plot = d3plot;

}


/*
 * Destructor.
 */
DB_Nodes::~DB_Nodes(){

  // Delete Nodes
  for (std::map<int,Node*>::iterator it=nodesByIndex.begin(); it!=nodesByIndex.end(); ++it){
    delete it->second;
  }

}


/*
 * Add a node to the db by node-ID and it's
 * coordinates vector. Returns a reference to the
 * new node.
 */
Node* DB_Nodes::add_node(int _nodeID, vector<float> coords){

  if(coords.size() != 3){
    throw("The node coordinate vector must have length 3.");
  }
  if(_nodeID < 0){
    throw("Node-ID may not be negative!");
  }

  // Check if node already is in map
  if(nodesByID.count(_nodeID) != 0)
    throw("Trying to insert a node with same id twice:"+to_string(_nodeID));

  // Create and add new node
  Node* node = new Node(_nodeID,coords,this);
  this->nodesByID.insert(pair<int,Node*>(_nodeID,node));
  this->nodesByIndex.insert(pair<int,Node*>(this->nodesByIndex.size()+1,node)); // starts at 1

  return node;

}


/*
 * Get a node from the node ID. If there is no
 * such node, NULL is returned.
 */
Node* DB_Nodes::get_nodeByID(int nodeID){

  map<int,Node*>::iterator it = this->nodesByID.find(nodeID);
  if(it == nodesByID.end()) return NULL;
  return it->second;

}


/*
 * Get a node from the node Index. If there is no
 * such node, NULL is returned. Indexes start at 1.
 */
Node* DB_Nodes::get_nodeByIndex(int nodeIndex){

  map<int,Node*>::iterator it =this->nodesByIndex.find(nodeIndex);
  if(it == nodesByIndex.end()) return NULL;
  return it->second;

}


/*
 * Register the element db in the node db.
 */
void DB_Nodes::set_db_elements(DB_Elements* _db_elements){
  if(_db_elements == NULL)
    throw("Setting db_elements=NULL in db_nodes is forbidden.");
  this->db_elements = _db_elements;
}

/*
 * Get the owning d3plot of the db.
 */
D3plot* DB_Nodes::get_d3plot(){
   return this->d3plot;
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
unsigned int DB_Nodes::size(){
  if(nodesByIndex.size() != nodesByID.size())
    throw("Node database encountered error: nodesByIndex.size() != nodesByID.size()");
  return nodesByIndex.size();
}
