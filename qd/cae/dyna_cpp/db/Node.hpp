
#ifndef NODE_HPP
#define NODE_HPP

// includes
#include <dyna_cpp/utility/PythonUtility.hpp>

#include <memory>
#include <vector>

namespace qd {

// forward declarations
class Element;
class DB_Nodes;

class Node
{
private:
  int32_t nodeID;
  std::vector<std::shared_ptr<Element>> elements;
  std::vector<float> coords;
  std::vector<std::vector<float>> disp;
  std::vector<std::vector<float>> vel;
  std::vector<std::vector<float>> accel;
  DB_Nodes* db_nodes;

public:
  Node(int32_t _nodeID, std::vector<float> _coords, DB_Nodes* db_nodes);
  ~Node();
  bool operator<(const Node& other) const;
  inline std::string str()
  {
    return "<Node id:" + std::to_string(nodeID) + ">";
  };

  std::shared_ptr<Element> add_element(std::shared_ptr<Element>);
  void add_disp(std::vector<float>);
  void add_vel(std::vector<float>);
  void add_accel(std::vector<float>);

  inline void clear_disp() { this->disp.clear(); }
  inline void clear_vel() { this->vel.clear(); }
  inline void clear_accel() { this->accel.clear(); }

  // Getter
  inline int32_t get_nodeID() { return this->nodeID; }
  inline std::vector<std::shared_ptr<Element>> get_elements()
  {
    return this->elements;
  }
  std::vector<float> get_coords(int32_t iTimestep = 0);
  inline std::vector<std::vector<float>> get_disp() { return this->disp; }
  inline std::vector<std::vector<float>> get_vel() { return this->vel; }
  inline std::vector<std::vector<float>> get_accel() { return this->accel; }
};

} // namespace qd

#endif
