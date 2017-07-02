
#include <iostream>
#include <stdexcept>
#include <string>

#include "dyna_cpp/db/DB_Elements.hpp"
#include "dyna_cpp/db/DB_Nodes.hpp"
#include "dyna_cpp/db/Element.hpp"
#include "dyna_cpp/db/FEMFile.hpp"
#include "dyna_cpp/db/Node.hpp"
#include "dyna_cpp/dyna/D3plot.hpp"
#include "dyna_cpp/utility/TextUtility.hpp"

using namespace std;

/*
 * Constructor.
 */
Node::Node(int _nodeID, vector<float> _coords, DB_Nodes* _db_nodes)
    : nodeID(_nodeID), coords(_coords), db_nodes(_db_nodes) {}

/*
 * Destructor.
 */
Node::~Node() {}

/*
 * Comparator.
 */
bool Node::operator<(const Node& other) const {
  return (this->nodeID < other.nodeID);
}

/*
 * Add an element to the node.
 */
Element* Node::add_element(Element* _element) {
#ifdef QD_DEBUG
  if (_element == nullptr)
    throw(std::invalid_argument("Trying to insert nullptr element to node:" +
                                to_string(this->nodeID)));
#endif

  this->elements.push_back(_element);
  return _element;
}

/*
 * Add a new displacement state to the node.
 */
void Node::add_disp(vector<float> new_disp) {
  if (new_disp.size() != 3)
    throw(std::invalid_argument(
        "Wrong length of displacement vector:" + to_string(new_disp.size()) +
        " in node:" + to_string(this->nodeID)));

  new_disp[0] -= coords[0];
  new_disp[1] -= coords[1];
  new_disp[2] -= coords[2];

  this->disp.push_back(new_disp);
}

/*
 * Add a new velocity state to the node.
 */
void Node::add_vel(vector<float> new_vel) {
#ifdef QD_DEBUG
  if (new_vel.size() != 3)
    throw(std::invalid_argument(
        "Wrong length of velocity vector:" + to_string(new_vel.size()) +
        " in node:" + to_string(this->nodeID)));
#endif

  this->vel.push_back(new_vel);
}

/*
 * Add a new velocity state to the node.
 */
void Node::add_accel(vector<float> new_accel) {
#ifdef QD_DEBUG
  if (new_accel.size() != 3)
    throw(std::invalid_argument(
        "Wrong length of velocity vector:" + to_string(new_accel.size()) +
        " in node:" + to_string(this->nodeID)));
#endif

  this->accel.push_back(new_accel);
}

/*
 * Get the coordinates of the node as an array
 * of length 3.
 */
vector<float> Node::get_coords(int iTimestep) {
  // D3plot with iTimestep != 0
  if (this->db_nodes->get_femfile()->is_d3plot()) {
    D3plot* d3plot = this->db_nodes->get_femfile()->get_d3plot();

    // Displacements iTimestep != 0
    if (iTimestep != 0) {
      if (d3plot->displacement_is_read()) {
        if (iTimestep < 0)
          iTimestep = static_cast<int>(d3plot->get_timesteps().size()) +
                      iTimestep;  // Python array style

        if ((iTimestep < 0))
          throw(std::invalid_argument(
              "Specified timestep exceeds real time step size."));

        if (iTimestep >= static_cast<long>(this->disp.size()))
          throw(std::invalid_argument(
              "Specified timestep exceeds real time step size."));

        vector<float> ret;
        ret = this->coords;  // copies

        ret[0] += this->disp[iTimestep][0];
        ret[1] += this->disp[iTimestep][1];
        ret[2] += this->disp[iTimestep][2];

        return ret;

      } else {
        throw(
            std::invalid_argument("Displacements were not read yet. Please use "
                                  "read_states=\"disp\"."));
      }
    }

    // KeyFile with iTimestep != 0
  } else if (this->db_nodes->get_femfile()->is_keyFile()) {
    if (iTimestep != 0)
      throw(std::invalid_argument(
          "Since a KeyFile has no states, you can not use the iTimeStep "
          "argument in node.get_coords."));

    // Unknown Filetype
  } else {
    throw(std::invalid_argument(
        "FEMFile is neither a d3plot, nor a keyfile in node.get_coords"));
  }

  // iTimestep == 0
  return this->coords;
}
