
#ifndef PART
#define PART

// forward declaration
class Node;
class Element;

#include <set>
#include <vector>
#include <iostream>
using namespace std;

class Part {

  private:
  int partID;
  string partName;
  set<Element*> elements;

  public:
  Part(int _partID,string _partName);
  ~Part();
  void set_name(string _partName);
  void add_element(Element* _element);

  int get_partID();
  string get_name();
  set<Node*> get_nodes();
  set<Element*> get_elements();

};

#endif
