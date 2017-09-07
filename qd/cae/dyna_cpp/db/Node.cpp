

#include "dyna_cpp/db/Node.hpp"
#include "dyna_cpp/db/DB_Elements.hpp"
#include "dyna_cpp/db/DB_Nodes.hpp"
#include "dyna_cpp/db/Element.hpp"
#include "dyna_cpp/db/FEMFile.hpp"
#include "dyna_cpp/dyna/D3plot.hpp"
#include "dyna_cpp/utility/TextUtility.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace qd {

/*
 * Constructor.
 */
Node::Node(int32_t _nodeID, std::vector<float> _coords, DB_Nodes* _db_nodes)
  : nodeID(_nodeID)
  , coords(_coords)
  , db_nodes(_db_nodes)
{
}

/*
 * Destructor.
 */
Node::~Node()
{
}

/*
 * Comparator.
 */
bool
Node::operator<(const Node& other) const
{
  return (this->nodeID < other.nodeID);
}

/*
 * Add an element to the node.
 */
std::shared_ptr<Element>
Node::add_element(std::shared_ptr<Element> _element)
{
#ifdef QD_DEBUG
  if (_element == nullptr)
    throw(std::invalid_argument("Trying to insert nullptr element to node:" +
                                std::to_string(this->nodeID)));
#endif

  this->elements.push_back(_element);
  return _element;
}

/*
 * Add a new displacement state to the node.
 */
void
Node::add_disp(std::vector<float> new_disp)
{
  if (new_disp.size() != 3)
    throw(std::invalid_argument(
      "Wrong length of displacement vector:" + std::to_string(new_disp.size()) +
      " in node:" + std::to_string(this->nodeID)));

  new_disp[0] -= coords[0];
  new_disp[1] -= coords[1];
  new_disp[2] -= coords[2];

  this->disp.push_back(new_disp);
}

/*
 * Add a new velocity state to the node.
 */
void
Node::add_vel(std::vector<float> new_vel)
{
#ifdef QD_DEBUG
  if (new_vel.size() != 3)
    throw(std::invalid_argument(
      "Wrong length of velocity vector:" + std::to_string(new_vel.size()) +
      " in node:" + std::to_string(this->nodeID)));
#endif

  this->vel.push_back(new_vel);
}

/*
 * Add a new velocity state to the node.
 */
void
Node::add_accel(std::vector<float> new_accel)
{
#ifdef QD_DEBUG
  if (new_accel.size() != 3)
    throw(std::invalid_argument(
      "Wrong length of velocity vector:" + std::to_string(new_accel.size()) +
      " in node:" + std::to_string(this->nodeID)));
#endif

  this->accel.push_back(new_accel);
}

/*
 * Get the coordinates of the node as an array
 * of length 3.
 */
std::vector<std::vector<float>>
Node::get_coords() const
{

  // check for displacements
  if (this->disp.size() > 0) {

    std::vector<std::vector<float>> ret(disp.size(), this->coords);
    for (size_t iTimestep = 0; iTimestep < this->disp.size(); ++iTimestep) {
      ret[iTimestep][0] += this->disp[iTimestep][0];
      ret[iTimestep][1] += this->disp[iTimestep][1];
      ret[iTimestep][2] += this->disp[iTimestep][2];
    }

    return ret;

    // no displacements
  } else {
    std::vector<std::vector<float>> ret = { this->coords };
    return ret;
  }
}

} // namespace qd