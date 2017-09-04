
#ifndef PART_HPP
#define PART_HPP

// includes
#include <dyna_cpp/db/Element.hpp>

#include <cstdint>
#include <vector>

namespace qd {

// forward declaration
class Node;
class FEMFile;

class Part
{
private:
  int32_t partID;
  FEMFile* femfile;
  std::string partName;
  std::vector<std::shared_ptr<Element>> elements;

public:
  explicit Part(int32_t _partID, std::string _partName, FEMFile* _femfile);
  ~Part();
  void set_name(std::string _partName);
  void add_element(std::shared_ptr<Element> _element);

  int32_t get_partID();
  std::string get_name();
  std::vector<std::shared_ptr<Node>> get_nodes();
  std::vector<std::shared_ptr<Element>> get_elements(
    Element::ElementType _etype = Element::NONE);
};

} // namespace qd

#endif
