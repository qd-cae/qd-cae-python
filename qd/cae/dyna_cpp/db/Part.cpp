
#include "Part.hpp"
#include "Node.hpp"
#include "Element.hpp"
#include "DB_Nodes.hpp"
#include "FEMFile.hpp"


/**
 * Constructor
 */
Part::Part(int _partID, string _partName, FEMFile *_femfile)
  : partName(_partName),
    partID( _partID ),
    femfile( _femfile ) {
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
  set<size_t> unique_node_indexes;

  // extract unique indexes
  for(vector<Element*>::iterator it=this->elements.begin(); it != this->elements.end(); ++it){
    vector<size_t> elem_node_indexes = ((Element*) *it)->get_node_indexes();
    std::copy( elem_node_indexes.begin(), 
               elem_node_indexes.end(), 
               std::inserter( unique_node_indexes, unique_node_indexes.end() ) );

  }

  // fetch nodes
  DB_Nodes* db_nodes = this->femfile->get_db_nodes();
  for( set<size_t>::const_iterator it=unique_node_indexes.begin();
       it != unique_node_indexes.end();
       ++it){

    nodes.push_back( db_nodes->get_nodeByIndex(*it) );
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

