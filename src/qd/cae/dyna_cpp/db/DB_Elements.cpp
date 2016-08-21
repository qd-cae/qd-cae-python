
#include <string>
#include <set>

#include "DB_Elements.hpp"
#include "DB_Nodes.hpp"
#include "DB_Parts.hpp"
#include "Element.hpp"
#include "Node.hpp"
#include "Part.hpp"
#include "FEMFile.hpp"
#include "../utility/TextUtility.hpp"

/** Constructor
 *
 * @param FEMFile* _femfile : parent file
 */
DB_Elements::DB_Elements(FEMFile* _femfile){

  this->femfile = _femfile;
  this->db_nodes = _femfile->get_db_nodes();
  this->db_parts = _femfile->get_db_parts();

}


/*
 * Destructor.
 */
DB_Elements::~DB_Elements(){

  // Delete
  for (std::map<int,Element*>::iterator it=elements2.begin(); it!=elements2.end(); ++it){
    delete it->second;
    it->second= NULL;
  }
  for (std::map<int,Element*>::iterator it=elements4.begin(); it!=elements4.end(); ++it){
    delete it->second;
    it->second= NULL;
  }
  for (std::map<int,Element*>::iterator it=elements8.begin(); it!=elements8.end(); ++it){
    delete it->second;
    it->second= NULL;
  }

}

/*
 * Add an element to the db by it's ID
 * and it's nodeIndexes. Throws an exception
 * if one nodeIndex is invalid or if the elementID
 * is already existing.
 */
Element* DB_Elements::add_element_byIndex(ElementType _eType, int _elementID, vector<int> _elementData){

  if(_elementID < 0){
    throw(string("Element-ID may not be negative!"));
  }

  // Find part
  Part* part = this->db_parts->get_part_byIndex(_elementData[_elementData.size()-1]);
  if(part == NULL){
    throw(string("Could not find part with index:")+to_string(_elementData[_elementData.size()-1])+string(" in db."));
  }

  // Find nodes
  set<Node*> nodes;
  for(size_t iNode = 0; iNode < _elementData.size()-1; iNode++){ // last is mat
    Node* _node = this->db_nodes->get_nodeByIndex(_elementData[iNode]);
    if(_node == NULL)
      throw(string("A node with index:")+to_string(_elementData[iNode])+string(" does not exist and can not be added to an element."));
    nodes.insert(_node);
  }

  // Create element
  Element* element = new Element(_elementID,_eType,nodes,this);
  //int _elementType = element->get_elementType();
  if(_eType == BEAM){
    map<int,Element*>::iterator it = this->elements2.find(_elementID);
    if(it != elements2.end()){
      delete element;
      throw(string("Trying to insert an element with same id twice:")+to_string(_elementID));
    }
    this->elements2.insert(pair<int,Element*>(_elementID,element));
    this->elements2ByIndex.insert(pair<int,Element*>(this->elements2ByIndex.size()+1,element));

  } else if(_eType == SHELL){
    map<int,Element*>::iterator it = this->elements4.find(_elementID);
    if(it != elements4.end()){
      delete element;
      throw(string("Trying to insert an element with same id twice:")+to_string(_elementID));
    }
    this->elements4.insert(pair<int,Element*>(_elementID,element));
    this->elements4ByIndex.insert(pair<int,Element*>(this->elements4ByIndex.size()+1,element));

  } else if(_eType == SOLID){
    map<int,Element*>::iterator it = this->elements8.find(_elementID);
    if(it != elements8.end()){
      delete element;
      throw(string("Trying to insert an element with same id twice:")+to_string(_elementID));
    }
    this->elements8.insert(pair<int,Element*>(_elementID,element));
    this->elements8ByIndex.insert(pair<int,Element*>(this->elements8ByIndex.size()+1,element));

  }

  // Register Elements
  //for(auto node : nodes) {
  for(set<Node*>::iterator it=nodes.begin(); it != nodes.end(); it++){
    ((Node*) *it)->add_element(element);
  }
  part->add_element(element);

  return element;
}


/** Add an element to the database coming from a KeyFile.
 * Add an element to the db by it's ID
 * and it's nodeIndexes. Since a KeyFile may have some weird order, missing
 * parts and nodes are created.
 */
Element* DB_Elements::add_element_byKeyFile(ElementType _eType,int _elementID, int _partid, vector<int> _node_ids)
{
  if(_elementID < 0){
    throw(string("Element-ID may not be negative!"));
  }

  // Find part
  Part* part = this->db_parts->get_part_byID(_partid);
  if(part == NULL){
     part = this->db_parts->add_part_byID(_partid);
  }

  // Find nodes
  set<Node*> nodes;
  for(size_t iNode = 0; iNode < _node_ids.size(); ++iNode){
    Node* _node = this->db_nodes->get_nodeByID(_node_ids[iNode]);
    if(_node == NULL)
      _node = this->db_nodes->add_node(_node_ids[iNode],vector<float>(3,0.0f));
    nodes.insert(_node);
  }

  // Create element
  Element* element = new Element(_elementID,_eType,nodes,this);
  //int _elementType = element->get_elementType();
  if(_eType == BEAM){
    map<int,Element*>::iterator it = this->elements2.find(_elementID);
    if(it != elements2.end()){
      delete element;
      throw(string("Trying to insert an element with same id twice:")+to_string(_elementID));
    }
    this->elements2.insert(pair<int,Element*>(_elementID,element));
    this->elements2ByIndex.insert(pair<int,Element*>(this->elements2ByIndex.size()+1,element));

  } else if(_eType == SHELL){
    map<int,Element*>::iterator it = this->elements4.find(_elementID);
    if(it != elements4.end()){
      delete element;
      throw(string("Trying to insert an element with same id twice:")+to_string(_elementID));
    }
    this->elements4.insert(pair<int,Element*>(_elementID,element));
    this->elements4ByIndex.insert(pair<int,Element*>(this->elements4ByIndex.size()+1,element));

  } else if(_eType == SOLID){
    map<int,Element*>::iterator it = this->elements8.find(_elementID);
    if(it != elements8.end()){
      delete element;
      throw(string("Trying to insert an element with same id twice:")+to_string(_elementID));
    }
    this->elements8.insert(pair<int,Element*>(_elementID,element));
    this->elements8ByIndex.insert(pair<int,Element*>(this->elements8ByIndex.size()+1,element));

  }

  // Register Elements
  //for(auto node : nodes) {
  for(set<Node*>::iterator it=nodes.begin(); it != nodes.end(); it++){
    ((Node*) *it)->add_element(element);
  }
  part->add_element(element);

  return element;
}


/** Get the element by it's internal id and it's type.
 *
 * @param int _elementType : see description
 * @param int _elementID : id of the element (like in the solver)
 * @return Element* _element
 *
 * Type may be: Element.ElementType
 * NONE = 0 -> error
 * BEAM = 1
 * SHELL = 2
 * SOLID = 3
 */
Element* DB_Elements::get_elementByID(int _elementType,int _elementID){

  if(_elementType == BEAM){
    map<int,Element*>::iterator it = this->elements2.find(_elementID);
    if(it == elements2.end())
      return NULL;
    return it->second;

  } else if(_elementType == SHELL){
    map<int,Element*>::iterator it = this->elements4.find(_elementID);
    if(it == elements4.end())
      return NULL;
    return it->second;

  } else if(_elementType == SOLID){
    map<int,Element*>::iterator it = this->elements8.find(_elementID);
    if(it == elements8.end())
      return NULL;
    return it->second;

  }

  throw("Can not get element with elementType:"+to_string(_elementID));

}

/** Get the element by it's internal index and it's type.
 *
 * @param int _elementType : see description
 * @param int _elementIndex : index of the element (not id!)
 * @return Element* _element
 *
 * Type may be: Element.ElementType
 * NONE = 0 -> error
 * BEAM = 1
 * SHELL = 2
 * SOLID = 3
 */
Element* DB_Elements::get_elementByIndex(int _elementType,int _elementIndex){

  if(_elementType == BEAM){
    map<int,Element*>::iterator it = this->elements2ByIndex.find(_elementIndex);
    if(it == elements2ByIndex.end())
      return NULL;
    return it->second;

  } else if(_elementType == SHELL){
    map<int,Element*>::iterator it = this->elements4ByIndex.find(_elementIndex);
    if(it == elements4ByIndex.end())
      return NULL;
    return it->second;

  } else if(_elementType == SOLID){
    map<int,Element*>::iterator it = this->elements8ByIndex.find(_elementIndex);
    if(it == elements8ByIndex.end())
      return NULL;
    return it->second;

  }

  throw("Can not get element with elementType:"+to_string(_elementIndex));

}


/** Get the DynaInputFile pointer
 * @return DnyaInputFile* keyfile
 */
FEMFile* DB_Elements::get_femfile(){
   return this->femfile;
}

/** Get the node-db.
 * @return DB_Nodes* db_nodes
 */
DB_Nodes* DB_Elements::get_db_nodes(){
  return this->db_nodes;
}


/** Get the number of  in the db.
 * @return unsigned int nElements : returns the total number of elements in the db
 */
size_t DB_Elements::size(){
  return elements4.size()+elements2.size()+elements8.size();
}
