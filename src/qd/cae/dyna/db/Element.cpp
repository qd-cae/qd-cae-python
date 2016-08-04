
#include <math.h> // sqrt
#include <algorithm> // std::max
#include <cmath>        // std::abs
#include <string>
#include "../dyna/d3plot.h"
#include "Element.h"
#include "Node.h"
#include "DB_Nodes.h"
#include "DB_Elements.h"
#include "../utility/TextUtility.h"
#include "../utility/MathUtility.h"

/*
 * Constructor.
 */
Element::Element(int _elementID, ElementType _elementType, set<Node*> _nodes,DB_Elements* _db_elements){

  // Checks
  if (_db_elements == NULL)
    throw("DB_Elements of an element may not be NULL in constructor.");

  // Assignment
  this->db_elements = _db_elements;
  this->elementID = _elementID;
  this->elemType = _elementType;
  
  for(set<Node*>::iterator it=_nodes.begin(); it != _nodes.end(); ++it){
    this->nodes.insert(((Node*) *it)->get_nodeID());
  }
  
  this->check();
  /*
  for (auto _node : _nodes)
    this->nodes.insert(_node->get_nodeID());
  */
}


/*
 * Destructor.
 */
Element::~Element(){

}


/*
 * Comparator.
 */
bool Element::operator<(const Element &other) const
{
  return(this->elementID < other.elementID);
}


/*
 * Get the element type of the element.
 *
 * NONE = 0
 * BEAM = 1
 * SHELL = 2
 * SOLID = 3
 */
ElementType Element::get_elementType(){
  return this->elemType;
}


/*
 * Get the elementID.
 */
int Element::get_elementID(){
  return this->elementID;
}


/*
 * Get the nodes of the element in a set.
 */
set<Node*> Element::get_nodes(){

  DB_Nodes* db_nodes = this->db_elements->get_db_nodes();
  set<Node*> node_set;

  for(set<int>::iterator it=nodes.begin(); it != nodes.end(); it++){

    Node* _node = db_nodes->get_nodeByID(*it);
    if(_node != NULL){
      node_set.insert(_node);
    } else{
      throw("Node with index:"+to_string(*it)+" in Element:"+to_string(this->elementID)+" was not found in DB.");
    }
        
  }
  
  return node_set;
}


/*
 * Append a value to the series of plastic strain.
 */
void Element::add_plastic_strain(float _platic_strain){

  /*
  if(_platic_strain < 0)
    throw("Element:"+to_string(this->elementID)+" tries to add a negative plastic strain:"+to_string(_platic_strain));
  */
    
  this->plastic_strain.push_back(_platic_strain);
}


/*
 * Append a value to the series of internal energy.
 */
void Element::add_energy(float _energy){

  if(_energy < 0)
    throw("Element:"+to_string(this->elementID)+" tries to add a negative energy:"+to_string(_energy));

  this->energy.push_back(_energy);
}


/*
 * Append a value to the series of strain.
 */
void Element::add_strain(vector<float> _strain){

  if(_strain.size() < 1)
    throw("Element:"+to_string(this->elementID)+" tries to add strain vector of length:"+to_string(_strain.size())+"!=6");

  this->strain.push_back(_strain);
}


/*
 * Append a value to the series of stress.
 */
void Element::add_stress(vector<float> _stress){

  if(_stress.size() < 1)
    throw("Element:"+to_string(this->elementID)+" tries to add stress vector of length:"+to_string(_stress.size())+"!=6");

  this->stress.push_back(_stress);
}

/*
 * Append history vars to the time seris.
 */
void Element::add_history_vars(vector<float> vars,size_t iTimestep){
   
   if(iTimestep < this->history_vars.size()){
      for(size_t ii=0; ii<vars.size(); ++ii){
         this->history_vars[iTimestep].push_back(vars[ii]);
      }
   } else {
      this->history_vars.push_back(vars);
   }
   
}


/*
 * Get the series of plastic strain. The
 * plastic strain here is accurately the
 * efficient plastic strain and thus a
 * single value, not a tensor.
 */
vector<float> Element::get_plastic_strain(){

  return this->plastic_strain;
}


/*
 * Get the series of the internal energy.
 */
vector<float> Element::get_energy(){

  return this->energy;
}


/*
 * Get the coordinates of the element, which is
 * the average of all nodes.
 */
vector<float> Element::get_coords(int iTimestep){
  
  if(this->nodes.size() < 1)
    throw("Element with id "+to_string(this->elementID)+" has no nodes and thus no coords.");
  
  if((iTimestep != 0) & (!this->db_elements->get_d3plot()->displacement_is_read()) ){
    throw(string("Displacements were not read yet. Please use read_states=\"disp\"."));
  }
  
  if( iTimestep < 0 )
    iTimestep = this->db_elements->get_d3plot()->get_timesteps().size() + iTimestep; // Python array style
  
  if( (iTimestep < 0) )
    throw(string("Specified timestep exceeds real time step size."));
  
  DB_Nodes* db_nodes = this->db_elements->get_db_nodes();
  
  Node* current_node = NULL;
  vector<float> coords_elem(3);
  vector<float> coords_node;
  vector< vector<float> > disp_node; 
  
  for(set<int>::iterator it=this->nodes.begin(); it != this->nodes.end(); ++it){
	
   current_node = db_nodes->get_nodeByID(*it);
   coords_node = current_node->get_coords();
	
   coords_elem[0] += coords_node[0];
   coords_elem[1] += coords_node[1];
   coords_elem[2] += coords_node[2];
    
   // Coords at different timestep
   if(iTimestep != 0){
      
      disp_node = current_node->get_disp();
      
      // Check correctness
      if( iTimestep >= disp_node.size() )
        throw(string("Specified timestep exceeds real time step size."));
      
      coords_elem[0] += disp_node[iTimestep][0];
      coords_elem[1] += disp_node[iTimestep][1];
      coords_elem[2] += disp_node[iTimestep][2];
   }
    
  }
  
  coords_elem[0] /= (float) this->nodes.size();
  coords_elem[1] /= (float) this->nodes.size();
  coords_elem[2] /= (float) this->nodes.size();
  
  return coords_elem;
	
} 

/*
 * Get an estimate for the average element length. This takes the 
 * maximum distance (diagonal) from the first node and multiplies
 * it with a volume factor (beam=1,shell=sqrt(2),solid=sqrt(3))
 */
float Element::get_estimated_element_size(){
   
   if(this->nodes.size() < 1)
      throw("Element with id "+to_string(this->elementID)+" has no nodes and thus no size.");
   
   DB_Nodes* db_nodes = this->db_elements->get_db_nodes();
   
   float maxdist = -1.;
   vector<float> ncoords;
   vector<float> basis_coords;
   for(set<int>::iterator it=this->nodes.begin(); it != this->nodes.end(); ++it){
      ncoords = db_nodes->get_nodeByID(*it)->get_coords();
      if(it != this->nodes.begin()){
         ncoords = MathUtility::v_subtr(ncoords,basis_coords);
         ncoords[0] *= ncoords[0];
         ncoords[1] *= ncoords[1];
         ncoords[2] *= ncoords[2];
         
         maxdist = max(maxdist,ncoords[0]+ncoords[1]+ncoords[2]);
      } else {
         basis_coords=ncoords;
      }
   }
   
   if(this->elemType == SHELL){
      if(this->nodes.size() == 3){
         return sqrt(maxdist); // tria
      } else if(this->nodes.size() ==  4){
         return sqrt(maxdist)/1.41421356237f; // quad
      } else {
         throw("Unknown node number:"+to_string(this->nodes.size())+" of element +"+to_string(this->elementID)+"+ for shells.");
      }
   }
   if(this->elemType == SOLID){
      if(this->nodes.size() == 4){
         return sqrt(maxdist); // tria
      } else if(this->nodes.size() == 8){
         return sqrt(maxdist)/1.73205080757f; // hexa
      } else if(this->nodes.size() ==  5){
         return sqrt(maxdist);  // pyramid ... difficult to handle
      } else if(this->nodes.size() ==  6){
         return sqrt(maxdist)/1.41421356237f; // penta
      } else {
         throw("Unknown node number:"+to_string(this->nodes.size())+" of element +"+to_string(this->elementID)+"+ for solids.");
      }
   }
   if(this->elemType == BEAM){
      if(this->nodes.size() != 2)
         throw("Unknown node number:"+to_string(this->nodes.size())+" of element +"+to_string(this->elementID)+"+ for beams.");
      return sqrt(maxdist); // beam
   }

   
}

/*
 * Get the series of strain. Strain is a vector
 * of 6 entries which represent the strain
 * tensor data.
 * e = [e_xx,e_yy,e_zz,e_xy,e_yz,e_xz]
 */
vector< vector<float> > Element::get_strain(){

  return this->strain;
}


/*
 * Get the series of stress. Stress is a vector
 * of 6 entries which represent the stress
 * tensor data.
 * s = [s_xx,s_yy,s_zz,s_xy,s_yz,s_xz]
 */
vector< vector<float> > Element::get_stress(){

  return this->stress;
}

/*
 * Get the series of history variables.
 */
vector< vector<float> > Element::get_history_vars(){

  return this->history_vars;
}

/*
 * Check of the element type is correct regarding the node size.
 */ 
void Element::check(){

   if(this->elemType == SHELL){
      if( (this->nodes.size() < 3) | (this->nodes.size() > 4))
         throw("A shell element must have 3 or 4 nodes. You have "+to_string(this->nodes.size()));
      return;
   }
   
   if(this->elemType == SOLID){
      if( (this->nodes.size() < 4) | (this->nodes.size() > 8) | (this->nodes.size() == 7))
         throw("A solid element must have 4,5,6 or 8 nodes. You have "+to_string(this->nodes.size()));
      return;
   }
   if(this->elemType == BEAM){
      if(this->nodes.size() != 2)
         throw("A beam element must have exactly 2 nodes. You have "+to_string(this->nodes.size()));
   }
   
}