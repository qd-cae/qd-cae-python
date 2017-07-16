
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/FEMFile.hpp>
#include <dyna_cpp/db/Node.hpp>
#include <dyna_cpp/db/Part.hpp>
#include <dyna_cpp/utility/TextUtility.hpp>

#include <set>

namespace qd {

/**
 * Constructor
 */
Part::Part(int32_t _partID, std::string _partName, FEMFile* _femfile)
  : partID(_partID)
  , femfile(_femfile)
  , partName(trim_copy(_partName))
{
}

/**
 * Destructor
 */
Part::~Part()
{
}

/**
 * Assign a part name.
 */
void
Part::set_name(std::string _name)
{
  this->partName = trim_copy(_name);
}

/**
 * Get the id of the part.
 */
int32_t
Part::get_partID()
{
  return this->partID;
}

/**
 * Get the name of the part.
 */
std::string
Part::get_name()
{
  return this->partName;
}

/**
 * Add a node to a part.
 */
void
Part::add_element(std::shared_ptr<Element> _element)
{
  this->elements.push_back(_element);
}

/**
 * Get the nodes of the part.
 */
std::vector<std::shared_ptr<Node>>
Part::get_nodes()
{
  std::vector<std::shared_ptr<Node>> nodes;
  std::set<size_t> unique_node_indexes;

  // extract unique indexes
  for (auto& elem : elements) {
    auto elem_node_indexes = elem->get_node_indexes();
    std::copy(elem_node_indexes.begin(),
              elem_node_indexes.end(),
              std::inserter(unique_node_indexes, unique_node_indexes.end()));
  }

  // fetch nodes
  DB_Nodes* db_nodes = this->femfile->get_db_nodes();
  for (const auto node_index : unique_node_indexes) {
    nodes.push_back(db_nodes->get_nodeByIndex(node_index));
  }

  return std::move(nodes);
}

/** Get the elements of the part.
 * @param Element::ElementType : optional filter
 * @return std::vector<Element*> elems
 */
std::vector<std::shared_ptr<Element>>
Part::get_elements(Element::ElementType _etype)
{
  if (_etype == Element::NONE) {
    return this->elements;

  } else {
    std::shared_ptr<Element> tmp_elem = nullptr;
    std::vector<std::shared_ptr<Element>> _elems;

    for (auto& tmp_elem : elements) {
      if (tmp_elem->get_elementType() == _etype) {
        _elems.push_back(tmp_elem);
      }
    }

    return _elems;
  }
}

} // namespace qd