
#ifndef NODE_HPP
#define NODE_HPP

// forward declarations
class Element;
class DB_Nodes;

// includes
#include <set>
#include <vector>
#include <iostream>

using namespace std;

class Node {

  private:
  int nodeID;
  set<Element*> elements;
  vector<float> coords;
  vector< vector<float> > disp;
  vector< vector<float> > vel;
  vector< vector<float> > accel;
  DB_Nodes* db_nodes;

  public:
  Node(int,vector<float>,DB_Nodes*);
  ~Node();
  bool operator<(const Node &other) const;
  Element* add_element(Element*);
  void add_disp (vector<float>);
  void add_vel  (vector<float>);
  void add_accel(vector<float>);
  void clear_disp();
  void clear_vel();
  void clear_accel();

  // Getter
  int get_nodeID();
  vector<float> get_coords(int iTimestep = 0);
  set<Element*> get_elements();
  vector< vector<float> > get_disp();
  vector< vector<float> > get_vel();
  vector< vector<float> > get_accel();


};

#endif
