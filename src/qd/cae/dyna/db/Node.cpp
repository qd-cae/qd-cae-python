
#include <string>
#include "Node.h"
#include "Element.h"
#include "DB_Nodes.h"
#include "DB_Elements.h"
#include "../dyna/d3plot.h"
#include "../utility/TextUtility.h"

/*
 * Constructor.
 */
Node::Node(int _nodeID, vector<float> _coords,DB_Nodes* _db_nodes){

  this->nodeID = _nodeID;
  this->coords = vector<float>(_coords); //copy
  this->elements = set<Element*>();
  this->db_nodes = _db_nodes;
}


/*
 * Destructor.
 */
Node::~Node(){

}


/*
 * Comparator.
 */
bool Node::operator<(const Node &other) const
{
  return(this->nodeID < other.nodeID);
}

/*
 * Add an element to the node.
 */
Element* Node::add_element(Element* _element){
  if(_element == NULL)
    throw("Trying to insert NULL element to node:"+to_string(this->nodeID));
  this->elements.insert(_element);
  return _element;
}


/*
 * Add a new displacement state to the node.
 */
void Node::add_disp(vector<float> new_disp){

  if(new_disp.size() != 3)
    throw("Wrong length of displacement vector:"+to_string(new_disp.size())+" in node:"+to_string(this->nodeID));

  for(int ii=0;ii<3;ii++)
    new_disp[ii] -= coords[ii];

  this->disp.push_back(new_disp);
}


/*
 * Add a new velocity state to the node.
 */
void Node::add_vel(vector<float> new_vel){

  if(new_vel.size() != 3)
    throw("Wrong length of velocity vector:"+to_string(new_vel.size())+" in node:"+to_string(this->nodeID));

  this->vel.push_back(new_vel);
}


/*
 * Add a new velocity state to the node.
 */
void Node::add_accel(vector<float> new_accel){

  if(new_accel.size() != 3)
    throw("Wrong length of velocity vector:"+to_string(new_accel.size())+" in node:"+to_string(this->nodeID));

  this->accel.push_back(new_accel);
}



/*
 * Get the node id. The node id is the id
 * given by the user in the input deck.
 */
int Node::get_nodeID(){
  return this->nodeID;
}


/*
 * Get the elements of the node.
 */
set<Element*> Node::get_elements(){
  return this->elements;
}



/*
 * Get the coordinates of the node as an array
 * of length 3.
 */
vector<float> Node::get_coords(int iTimestep){
  
  if((iTimestep != 0) & (!this->db_nodes->get_d3plot()->displacement_is_read()) ){
    throw(string("Displacements were not read yet. Please use read_states=\"disp\"."));
  }

  if( iTimestep < 0 )
    iTimestep = this->db_nodes->get_d3plot()->get_timesteps().size() + iTimestep; // Python array style
  
  if( (iTimestep < 0) )
    throw(string("Specified timestep exceeds real time step size."));
  
  if(iTimestep == 0){
     
     return this->coords;
  
  } else {
     
     vector<float> ret;
     ret = this->coords;
     
     if( iTimestep >= this->disp.size() )
        throw(string("Specified timestep exceeds real time step size."));
     
     ret[0] += this->disp[iTimestep][0];
     ret[1] += this->disp[iTimestep][1];
     ret[2] += this->disp[iTimestep][2];
     
     return ret;
  }
  
}


/*
 * Get the coordinates of the node as an array
 * of length 3.
 */
vector< vector<float> > Node::get_disp(){
  return this->disp;
}


/*
 * Get the coordinates of the node as an array
 * of length 3.
 */
vector< vector<float> > Node::get_vel(){
  return this->vel;
}


/*
 * Get the coordinates of the node as an array
 * of length 3.
 */
vector< vector<float> > Node::get_accel(){
  return this->accel;
}
