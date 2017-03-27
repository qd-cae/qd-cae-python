
#ifndef PART_HPP
#define PART_HPP

// forward declaration
class Node;
class FEMFile;
//class Element;

#include <vector>
#include "dyna_cpp/db/Element.hpp"

class Part {

  private:
  int partID;
  FEMFile *femfile;
  std::string partName;
  std::vector<Element*> elements;

  public:
  Part(int _partID, std::string _partName, FEMFile* _femfile);
  ~Part();
  void set_name(std::string _partName);
  void add_element(Element* _element);

  int get_partID();
  std::string get_name();
  std::vector<Node*> get_nodes();
  std::vector<Element*> get_elements(ElementType _etype = NONE);

};

#endif
