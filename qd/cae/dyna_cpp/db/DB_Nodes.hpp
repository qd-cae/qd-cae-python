
#ifndef DB_NODES_HPP
#define DB_NODES_HPP

// forward declarations
class Node;
class FEMFile;
class DB_Elements;

// includes
#include <unordered_map>
#include <memory>
#include <vector>
#include <string>
#include "dyna_cpp/utility/TextUtility.hpp"

class DB_Nodes {

private:
  FEMFile* femfile;
  std::unordered_map<int,size_t> id2index;
  std::vector< std::unique_ptr<Node> > nodes;
  DB_Elements* db_elements;

public:
  DB_Nodes(FEMFile* _femfile);
  ~DB_Nodes();
  size_t size();
  void reserve(const size_t _size);
  FEMFile* get_femfile();
  DB_Elements* get_db_elements();
  void set_db_elements(DB_Elements*);
  Node* add_node(int _id, std::vector<float> _coords);

  template<typename T>
  T get_id_from_index(size_t _id);
  template<typename T>
  size_t get_index_from_id(T _index);

  template<typename T>
  Node* get_nodeByID(T _id);
  template<typename T>
  Node* get_nodeByIndex(T _index);

};



/** Get the node index from it's id
 *
 * @param T _id : node id
 * @return size_t : node index
 */
template<typename T>
T DB_Nodes::get_id_from_index(size_t _index){

  static_assert(std::is_integral<T>::value, "Integer number required.");

  if(_index > nodes.size()-1)
    throw( std::string("Node with index ")+to_string(_index)+std::string(" does not exist in the db.") );
  return _index;

}


/** Get the node id from it's index
 *
 * @param T _id : node id
 * @return size_t _index : node index
 */
template<typename T>
size_t DB_Nodes::get_index_from_id(T _id){

  static_assert(std::is_integral<T>::value, "Integer number required.");

  const auto& it = this->id2index.find(_id);
  if(it == this->id2index.end())
    throw( std::string("Node with id ")+to_string(_id)+std::string(" does not exist in the db.") );
  return it->second;

}


/** Get a node from the node ID.
 *
 * @param T _id : id of the node
 * @return Node* node : pointer to the node or nullptr if node is not existing!
 */
template <typename T>
inline Node* DB_Nodes::get_nodeByID(T _id){

  static_assert(std::is_integral<T>::value, "Integer number required.");

  const auto& it = this->id2index.find(_id);
  if(it == this->id2index.end())
    return nullptr;
  return this->nodes[ it->second ].get();

}


/** Get a node from the node index.
 *
 * @param int _nodeIndex : index of the node
 * @return Node* node : pointer to the node or nullptr if node is not existing!
 */
template <typename T>
inline Node* DB_Nodes::get_nodeByIndex(T _nodeIndex){

  static_assert(std::is_integral<T>::value, "Integer number required.");

  if(_nodeIndex >= this->nodes.size())
     return nullptr;
  return this->nodes[_nodeIndex].get();

}

#endif
