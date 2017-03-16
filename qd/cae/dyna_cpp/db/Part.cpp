
#include "Part.hpp"
#include "Node.hpp"
#include "Element.hpp"


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
  this->elements.push_back(_element);
}


/**
 * Get the nodes of the part.
 */
vector<Node*> Part::get_nodes(){

  vector<Node*> nodes;
  vector<Node*> elem_nodes;

  for(vector<Element*>::iterator it=this->elements.begin(); it != this->elements.end(); ++it){
    elem_nodes = ((Element*) *it)->get_nodes();
    for(vector<Node*>::iterator it2=elem_nodes.begin(); it2 != elem_nodes.end(); ++it2){
	   nodes.push_back((Node*) *it2);
    }
  }

  return nodes;
}


/**
 * Get the elements of the part.
 */
vector<Element*> Part::get_elements(ElementType _etype){

  if(_etype == NONE){
    return this->elements;

  } else {
    
    Element* tmp_elem = NULL;
    vector<Element*> _elems;

    for(vector<Element*>::iterator it=this->elements.begin(); it != this->elements.end(); ++it ){

      tmp_elem = (Element*) *it;
      if( tmp_elem->get_elementType() == _etype ){
        _elems.push_back(tmp_elem);
      }
    }

    return _elems;
  }
  
}

