
#ifndef ELEMENT
#define ELEMENT

// forward declarations
class Node;
class DB_Elements;

// includes
#include <set>
#include <vector>
#include <iostream>

using namespace std;

// enum
enum ElementType {NONE,BEAM,SHELL,SOLID};

class Element {

  /* PRIVATE */
  private:

  int elementID;
  set<int> nodes;
  vector<float> energy;
  vector<float> plastic_strain;
  vector< vector<float> > strain;
  vector< vector<float> > stress;
  vector< vector<float> > history_vars;
  ElementType elemType;
  DB_Elements* db_elements;

  /* PUBLIC */
  public:
  Element(int,ElementType,set<Node*>,DB_Elements* db_elements);
  ~Element();
  bool operator<(const Element &other) const;
  void check();

  // getter
  ElementType   get_elementType();
  int           get_elementID();
  float         get_estimated_element_size(); // fast
  set<Node*>    get_nodes();
  vector<float> get_coords(int iTimestep = 0);
  vector<float> get_energy();
  vector<float> get_plastic_strain();
  vector< vector<float> > get_strain();
  vector< vector<float> > get_stress();
  vector< vector<float> > get_history_vars();

  // setter
  void add_energy(float);
  void add_plastic_strain(float);
  void add_stress(vector<float>);
  void add_strain(vector<float>);
  void add_history_vars(vector<float> vars,size_t iTimestep);

};

#endif
