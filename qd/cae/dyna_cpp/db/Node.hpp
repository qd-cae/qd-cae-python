
#ifndef NODE_HPP
#define NODE_HPP

// forward declarations
class Element;
class DB_Nodes;

// includes
#include <vector>

class Node {

  private:
  int nodeID;
  std::vector<Element*> elements;
  std::vector<float> coords;
  std::vector< std::vector<float> > disp;
  std::vector< std::vector<float> > vel;
  std::vector< std::vector<float> > accel;
  DB_Nodes* db_nodes;

  public:
  Node(int _nodeID, 
       std::vector<float> _coords, 
       DB_Nodes* db_nodes);
  ~Node();
  bool operator<(const Node &other) const;
  Element* add_element(Element*);
  void add_disp (std::vector<float>);
  void add_vel  (std::vector<float>);
  void add_accel(std::vector<float>);
  inline void clear_disp(){this->disp.clear();}
  inline void clear_vel(){this->vel.clear();}
  inline void clear_accel(){this->accel.clear();}

  // Getter
  inline int get_nodeID(){return this->nodeID;}
  inline std::vector<Element*> get_elements(){return this->elements;}
  std::vector<float> get_coords(int iTimestep = 0);
  inline std::vector< std::vector<float> > get_disp(){return this->disp;}
  inline std::vector< std::vector<float> > get_vel(){return this->vel;}
  inline std::vector< std::vector<float> > get_accel(){return this->accel;}


};

#endif
