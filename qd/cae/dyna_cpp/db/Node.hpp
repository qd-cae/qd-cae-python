
#ifndef NODE_HPP
#define NODE_HPP

// includes
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace qd {

// forward declarations
class Element;
class DB_Nodes;
class DB_Elements;

class Node
{
  friend class DB_Elements;

private:
  int32_t nodeID;
  std::vector<std::shared_ptr<Element>> elements;
  std::vector<float> coords;
  std::vector<std::vector<float>> disp;
  std::vector<std::vector<float>> vel;
  std::vector<std::vector<float>> accel;
  DB_Nodes* db_nodes;

  std::mutex _node_mutex;

  void remove_element(std::shared_ptr<Element> _element);

public:
  explicit Node(int32_t _nodeID,
                const std::vector<float>& _coords,
                DB_Nodes* db_nodes);
  explicit Node(int32_t _nodeID,
                float _x,
                float _y,
                float _z,
                DB_Nodes* db_nodes);
  virtual ~Node();
  bool operator<(const Node& other) const;
  inline std::string str()
  {
    return "<Node id:" + std::to_string(nodeID) + ">";
  };

  std::shared_ptr<Element> add_element(std::shared_ptr<Element>);
  void add_disp(std::vector<float>);
  void add_vel(std::vector<float>);
  void add_accel(std::vector<float>);
  void set_coords(float _x, float _y, float _z);

  inline void clear_disp();
  inline void clear_vel();
  inline void clear_accel();

  // Getter
  inline int32_t get_nodeID() const;
  inline std::vector<std::shared_ptr<Element>> get_elements();

  inline const std::vector<float>& get_position() const;
  inline std::vector<std::vector<float>> get_coords() const;
  inline const std::vector<std::vector<float>>& get_disp() const;
  inline const std::vector<std::vector<float>>& get_vel() const;
  inline const std::vector<std::vector<float>>& get_accel() const;
};

/** Clear the displacements
 *
 */
void
Node::clear_disp()
{
  std::lock_guard<std::mutex> lock(_node_mutex);
  this->disp.clear();
}

/** Clear the velocity
 *
 */
void
Node::clear_vel()
{
  std::lock_guard<std::mutex> lock(_node_mutex);
  this->vel.clear();
}

/** Clear the acceleration
 *
 */
void
Node::clear_accel()
{
  std::lock_guard<std::mutex> lock(_node_mutex);
  this->accel.clear();
}

/** Get the external id of the node
 *
 * @return id
 */
int32_t
Node::get_nodeID() const
{
  return this->nodeID;
}

/** Get all elements of the node
 *
 * @return elements
 */
std::vector<std::shared_ptr<Element>>
Node::get_elements()
{
  return elements;
}

/** Get the position of the node
 *
 * @return position
 */
inline const std::vector<float>&
Node::get_position() const
{
  return coords;
}

/** Get the coordinates of the node over time
 *
 * @return ret : time series of coordinates
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

/** Get the displacement over time of the node
 *
 * @return displacement
 */
const std::vector<std::vector<float>>&
Node::get_disp() const
{
  return this->disp;
}

/** Get the velocity over time of the node
 *
 * @return velocity
 */
const std::vector<std::vector<float>>&
Node::get_vel() const
{
  return this->vel;
}

/** Get the acceleration over time of the node
 *
 * @return acceleration
 */
const std::vector<std::vector<float>>&
Node::get_accel() const
{
  return this->accel;
}

} // namespace qd

#endif
