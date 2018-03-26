
#ifndef PART_HPP
#define PART_HPP

// includes
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>


#include <dyna_cpp/db/Element.hpp>

namespace qd {

// forward declaration
class Node;
class FEMFile;
class DB_Elements;

class Part
{
  friend class DB_Elements;

private:
  int32_t partID;
  FEMFile* femfile;
  std::string partName;
  std::vector<std::shared_ptr<Element>> elements;

  std::mutex _part_mutex;

  void remove_element(std::shared_ptr<Element> _element);

public:
  explicit Part(int32_t _partID,
                const std::string& _partName,
                FEMFile* _femfile);
  void set_name(const std::string& _partName);
  void add_element(std::shared_ptr<Element> _element);

  int32_t get_partID() const;
  std::string get_name() const;
  std::vector<std::shared_ptr<Node>> get_nodes();
  std::vector<std::shared_ptr<Element>> get_elements(
    Element::ElementType _etype = Element::NONE);
};

} // namespace qd

#endif
