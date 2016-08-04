
#include <string>
#include <set>

#include "DB_Elements.h"
#include "DB_Nodes.h"
#include "DB_Parts.h"
#include "Element.h"
#include "Node.h"
#include "Part.h"
#include "../dyna/d3plot.h"
#include "../utility/TextUtility.h"

/*
 * Constructor.
 */
DB_Elements::DB_Elements(D3plot* _d3plot, DB_Nodes* _db_nodes, DB_Parts* _db_parts){

  this->d3plot = _d3plot;
  this->db_nodes = _db_nodes;
  this->db_parts = _db_parts;

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
Element* DB_Elements::add_element(ElementType _eType, int _elementID, vector<int> _elementData){

  if(_elementID < 0){
    throw("Element-ID may not be negative!");
  }

  // Find part
  Part* part = this->db_parts->get_part_byIndex(_elementData[_elementData.size()-1]);
  if(part == NULL){
    throw("Could not find part with index:"+to_string(_elementData[_elementData.size()-1])+" in db.");
  }

  // Find nodes
  set<Node*> nodes;
  for(unsigned int iNode = 0; iNode < _elementData.size()-1; iNode++){ // last is mat
    Node* _node = this->db_nodes->get_nodeByIndex(_elementData[iNode]);
    if(_node == NULL)
      throw("A node with index:"+to_string(_elementData[iNode])+" does not exist and can not be added to an element.");
    nodes.insert(_node);
  }

  // Create element
  Element* element = new Element(_elementID,_eType,nodes,this);
  //int _elementType = element->get_elementType();
  if(_eType == BEAM){
    map<int,Element*>::iterator it = this->elements2.find(_elementID);
    if(it != elements2.end()){
      delete element;
      throw("Trying to insert an element with same id twice:"+to_string(_elementID));
    }
    this->elements2.insert(pair<int,Element*>(_elementID,element));
    this->elements2ByIndex.insert(pair<int,Element*>(this->elements2ByIndex.size()+1,element));

  } else if(_eType == SHELL){
    map<int,Element*>::iterator it = this->elements4.find(_elementID);
    if(it != elements4.end()){
      delete element;
      throw("Trying to insert an element with same id twice:"+to_string(_elementID));
    }
    this->elements4.insert(pair<int,Element*>(_elementID,element));
    this->elements4ByIndex.insert(pair<int,Element*>(this->elements4ByIndex.size()+1,element));

  } else if(_eType == SOLID){
    map<int,Element*>::iterator it = this->elements8.find(_elementID);
    if(it != elements8.end()){
      delete element;
      throw("Trying to insert an element with same id twice:"+to_string(_elementID));
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


/*
 * Get the element by it's id and it's type.
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

/*
 * Get the element by it's internal index and it's type.
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

/*
 * Get the d3plot
 */
D3plot* DB_Elements::get_d3plot(){
   return this->d3plot;
}
 

/*
 * Get the node-db.
 */
DB_Nodes* DB_Elements::get_db_nodes(){
  return this->db_nodes;
}


/*
 * Get the number of  in the db.
 */
unsigned int DB_Elements::size(){
  return elements4.size()+elements2.size()+elements8.size();
}
