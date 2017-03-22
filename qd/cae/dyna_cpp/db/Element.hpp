
#ifndef ELEMENT_HPP
#define ELEMENT_HPP

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
  bool is_rigid;
  vector<size_t> nodes; // indexes
  vector<float> energy;
  vector<float> stress_mises;
  vector<float> plastic_strain;
  vector< vector<float> > strain;
  vector< vector<float> > stress;
  vector< vector<float> > history_vars;
  ElementType elemType;
  DB_Elements* db_elements;

  /* PUBLIC */
public:
  Element(const int _id, const ElementType _etype, const vector<size_t>& _nodes, DB_Elements* db_elements);
  ~Element();
  bool operator<(const Element &other) const;
  void check() const;

  // getter
  ElementType   get_elementType() const;
  int           get_elementID() const;
  bool          get_is_rigid() const;
  float         get_estimated_element_size() const; // fast
  vector<Node*>    get_nodes() const;
  vector<int>      get_node_ids() const;
  vector<size_t>   get_node_indexes() const;
  vector<float> get_coords(int iTimestep = 0) const;
  vector<float> get_energy() const;
  vector<float> get_stress_mises() const;
  vector<float> get_plastic_strain() const;
  vector< vector<float> > get_strain() const;
  vector< vector<float> > get_stress() const;
  vector< vector<float> > get_history_vars() const;

  // setter
  void set_is_rigid(bool _is_rigid);
  void add_energy(float _energy);
  void add_stress_mises(float _stress_mises);
  void add_plastic_strain(float _plastic_strain);
  void add_stress(vector<float> _stress);
  void add_strain(vector<float> _strain);
  void add_history_vars(vector<float> vars, size_t iTimestep);

  // clearer
  void clear_energy();
  void clear_plastic_strain();
  void clear_stress();
  void clear_stress_mises();
  void clear_strain();
  void clear_history_vars();

};

#endif
