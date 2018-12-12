

#include "dyna_cpp/db/Node.hpp"
#include "dyna_cpp/db/DB_Elements.hpp"
#include "dyna_cpp/db/DB_Nodes.hpp"
#include "dyna_cpp/db/Element.hpp"
#include "dyna_cpp/db/FEMFile.hpp"
#include "dyna_cpp/utility/TextUtility.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace qd {

/** Constructor of a node
 * @param _nodeID : id of the node
 * @param _coords : coordinate vector, must have length 3
 * @param _db_nodes : parent database
 */
Node::Node(int32_t _nodeID,
           const std::vector<float>& _coords,
           DB_Nodes* _db_nodes)
  : nodeID(_nodeID)
  , coords(_coords)
  , db_nodes(_db_nodes)
{}

/** Constructor of a node
 * @param _x
 * @param _y
 * @param _z
 * @param _db_nodes : pointer to owning database
 */
Node::Node(int32_t _nodeID, float _x, float _y, float _z, DB_Nodes* _db_nodes)
  : nodeID(_nodeID)
  , coords({ _x, _y, _z })
  , db_nodes(_db_nodes)
{}

/** Destructor
 *
 */
Node::~Node()
{
#ifdef QD_DEBUG
// std::cout << "Node " << nodeID << " erased." << '\n';
#endif
}

/** Comparator.
 * Used for maps ... somwhere ... can not remember
 */
bool
Node::operator<(const Node& other) const
{
  return (this->nodeID < other.nodeID);
}

/** Add an element to the node.
 * @param _element : element to add to the node
 * @return _element : returns argument element
 */
std::shared_ptr<Element>
Node::add_element(std::shared_ptr<Element> _element)
{
#ifdef QD_DEBUG
  if (_element == nullptr)
    throw(std::invalid_argument("Trying to insert nullptr element to node:" +
                                std::to_string(this->nodeID)));
#endif

  {
    std::lock_guard<std::mutex> lock(_node_mutex);
    this->elements.push_back(_element);
  }

  return _element;
}

/** Add a new displacement state to the node.
 *
 * @param _new_disp : new displacement vector
 */
void
Node::add_disp(std::vector<float> _new_disp)
{
#ifdef QD_DEBUG
  if (_new_disp.size() != 3)
    throw(std::invalid_argument("Wrong length of displacement vector:" +
                                std::to_string(_new_disp.size()) +
                                " in node:" + std::to_string(this->nodeID)));
#endif

  _new_disp[0] -= coords[0];
  _new_disp[1] -= coords[1];
  _new_disp[2] -= coords[2];

  {
    std::lock_guard<std::mutex> lock(_node_mutex);
    this->disp.push_back(_new_disp);
  }
}

/** Add a new velocity state to the node.
 *
 * @param _new_vel : new vector
 */
void
Node::add_vel(std::vector<float> _new_vel)
{
#ifdef QD_DEBUG
  if (_new_vel.size() != 3)
    throw(std::invalid_argument(
      "Wrong length of velocity vector:" + std::to_string(_new_vel.size()) +
      " in node:" + std::to_string(this->nodeID)));
#endif
  {
    std::lock_guard<std::mutex> lock(_node_mutex);
    this->vel.push_back(_new_vel);
  }
}

/** Add a new velocity state to the node.
 *
 * @param _new_accel : new vector
 */
void
Node::add_accel(std::vector<float> _new_accel)
{
#ifdef QD_DEBUG
  if (_new_accel.size() != 3)
    throw(std::invalid_argument(
      "Wrong length of velocity vector:" + std::to_string(_new_accel.size()) +
      " in node:" + std::to_string(this->nodeID)));
#endif
  {
    std::lock_guard<std::mutex> lock(_node_mutex);
    this->accel.push_back(_new_accel);
  }
}

/** Set the coordinates of the node
 * @param _x
 * @param _y
 * @param _z
 */
void
Node::set_coords(float _x, float _y, float _z)
{
  coords[0] = _x;
  coords[1] = _y;
  coords[2] = _z;
}

/** Remove an element from the node
 *
 * @param _element : element to remove
 *
 * Does nothing if element referenced by node
 */
void
Node::remove_element(std::shared_ptr<Element> _element)
{
  {
    std::lock_guard<std::mutex> lock(_node_mutex);
    elements.erase(std::remove_if(elements.begin(),
                                  elements.end(),
                                  [_element](std::shared_ptr<Element> elem) {
                                    return elem == _element;
                                  }),
                   elements.end());
  }
}

} // NAMESPACE:qd