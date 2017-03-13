
#ifndef DB_NODES_HPP
#define DB_NODES_HPP

// forward declarations
class Node;
class FEMFile;
class DB_Elements;

// includes
#include <map>
#include <vector>

// namespaces
using namespace std;

class DB_Nodes {

private:
  FEMFile* femfile;
  map<int,size_t> id2index;
  vector<Node*> nodes;
  DB_Elements* db_elements;

public:
  DB_Nodes(FEMFile* _femfile);
  ~DB_Nodes();
  size_t size();
  void reserve(const size_t _size);
  FEMFile* get_femfile();
  DB_Elements* get_db_elements();
  void set_db_elements(DB_Elements*);
  Node* add_node(int _id,vector<float> _coords);

  template<typename T>
  Node* get_nodeByID(T _id);
  template<typename T>
  Node* get_nodeByIndex(T _index);

};

/** Get a node from the node ID.
 *
 * @param int _nodeID : id of the node
 * @return Node* node : pointer to the node or NULL if node is not existing!
 */
template <typename T>
Node* DB_Nodes::get_nodeByID(T nodeID){

  map<int,size_t>::iterator it = this->id2index.find(nodeID);
  if(it == this->id2index.end())
    return NULL;
  return this->nodes[it->second];

}


/** Get a node from the node index.
 *
 * @param int _nodeIndex : index of the node
 * @return Node* node : pointer to the node or NULL if node is not existing!
 */
template <typename T>
Node* DB_Nodes::get_nodeByIndex(T _nodeIndex){

  if(_nodeIndex >= this->nodes.size())
     return NULL;
  return this->nodes[_nodeIndex];

}

#endif
