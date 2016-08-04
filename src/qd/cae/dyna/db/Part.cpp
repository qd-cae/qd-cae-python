
#include "Part.h"
#include "Node.h"
#include "Element.h"


/**
 * Constructor
 */
Part::Part(int _partID,string _partName){

  this->partName = _partName;
  this->partID = _partID;

}


/**
 * Destructor
 */
Part::~Part(){

}

/**
 * Assign a part name.
 */
void Part::set_name(string _name){
  
  string::const_iterator it = _name.begin();
  while (it != _name.end() && isspace(*it))
    it++;
	
  string::const_reverse_iterator rit = _name.rbegin();
  while (rit.base() != it && isspace(*rit))
    rit++;

  string name(it, rit.base());
	
  this->partName = name;
}


/**
 * Get the id of the part.
 */
int Part::get_partID(){
  return this->partID;
}


/**
 * Get the name of the part.
 */
string Part::get_name(){
  return this->partName;
}


/**
 * Add a node to a part.
 */
void Part::add_element(Element* _element){
  this->elements.insert(_element);
}


/**
 * Get the nodes of the part.
 */
set<Node*> Part::get_nodes(){

  set<Node*> nodes;
  set<Node*> elem_nodes;
  
  for(set<Element*>::iterator it=this->elements.begin(); it != this->elements.end(); ++it){
    elem_nodes = ((Element*) *it)->get_nodes();
    for(set<Node*>::iterator it2=elem_nodes.begin(); it2 != elem_nodes.end(); ++it2){
	   nodes.insert((Node*) *it2);
    }
  }
  /*
  for(auto elem : this->elements){
    for(auto node : elem->get_nodes()){
      nodes.insert(node);
    }
  }
  */
  return nodes;
}


/**
 * Get the elements of the part.
 */
set<Element*> Part::get_elements(){
  return this->elements;
}
