
#ifndef PART_HPP
#define PART_HPP

// forward declaration
class Node;
//class Element;

#include <set>
#include <vector>
#include <iostream>
#include "Element.hpp"
using namespace std;

class Part {

  private:
  int partID;
  string partName;
  vector<Element*> elements;

  public:
  Part(int _partID,string _partName);
  ~Part();
  void set_name(string _partName);
  void add_element(Element* _element);

  int get_partID();
  string get_name();
  vector<Node*> get_nodes();
  vector<Element*> get_elements(ElementType _etype = NONE);

};

#endif
