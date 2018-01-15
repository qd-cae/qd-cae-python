
#include <algorithm>
#include <unordered_set>

#include "DB_Elements.hpp"
#include "DB_Nodes.hpp"
#include "DB_Parts.hpp"
#include "Element.hpp"
#include "FEMFile.hpp"
#include "Node.hpp"
#include "Part.hpp"

namespace qd {

/** Constructor
 *
 * @param FEMFile* _femfile : parent file
 */
DB_Elements::DB_Elements(FEMFile* _femfile)
  : femfile(_femfile)
  , db_nodes(_femfile->get_db_nodes())
  , db_parts(_femfile->get_db_parts())
{}

/*
 * Destructor.
 */
DB_Elements::~DB_Elements()
{
#ifdef QD_DEBUG
  std::cout << "DB_Elements::~DB_Elements called." << std::endl;
#endif
}

/** Add an element to the database (internal usage only)
 *
 * @param _etype : type of the element
 * @param _elementID : id of the element
 * @param _node_indexes : node indexes in db (must be in same db!!!)
 *
 * Adds an element, will not perform checks that the element
 * has the same database as nodes and part.
 * Also not checked whether the nodes with the specified indexes
 * exist.
 */
std::shared_ptr<Element>
DB_Elements::create_element_unchecked(Element::ElementType _eType,
                                      int32_t _id,
                                      const std::vector<size_t>& _node_indexes)
{
  std::shared_ptr<Element> element =
    std::make_shared<Element>(_id, _eType, _node_indexes, this);

  switch (_eType) {
    case (Element::SHELL):
      // check uniqueness
      if (id2index_elements4.find(_id) != id2index_elements4.end())
        throw(std::invalid_argument(
          "Trying to insert an element with same id twice:" +
          std::to_string(_id)));
      // insert
      id2index_elements4.insert(
        std::pair<int32_t, size_t>(_id, elements4.size()));
      elements4.push_back(element);
      break;

    case (Element::SOLID):
      if (id2index_elements8.find(_id) != id2index_elements8.end())
        throw(std::invalid_argument(
          "Trying to insert an element with same id twice:" +
          std::to_string(_id)));

      id2index_elements8.insert(
        std::pair<int32_t, size_t>(_id, elements8.size()));
      elements8.push_back(element);
      break;

    case (Element::BEAM):
      if (id2index_elements2.find(_id) != id2index_elements2.end())
        throw(std::invalid_argument(
          "Trying to insert an element with same id twice:" +
          std::to_string(_id)));

      this->id2index_elements2.insert(
        std::pair<int32_t, size_t>(_id, elements2.size()));
      this->elements2.push_back(element);
      break;

    case (Element::TSHELL):
      if (id2index_elements4th.find(_id) != id2index_elements4th.end())
        throw(std::invalid_argument(
          "Trying to insert an element with same id twice:" +
          std::to_string(_id)));

      id2index_elements4th.insert(
        std::pair<int32_t, size_t>(_id, this->elements4th.size()));
      elements4th.push_back(element);
      break;

    default:
      throw(std::invalid_argument(
        "Element with an invalid element type was tried to get inserted "
        "into the database."));
      break;
  }

  return element;
}

/** Add an element to the database from node indexes
 *
 * @param _etype : type of the element
 * @param _elementID : id of the element
 * @param _part_id : id of the part the element belongs to
 * @param _node_indexes : indexes of nodes
 */
std::shared_ptr<Element>
DB_Elements::add_elementByNodeIndex(const Element::ElementType _eType,
                                    int32_t _elementID,
                                    int32_t _part_id,
                                    const std::vector<size_t>& _node_indexes)
{
  if (_elementID < 0) {
    throw(std::invalid_argument("Element-ID may not be negative!"));
  }

  // Find part
  const auto part = db_parts->get_partByID(_part_id);
  if (part == nullptr) {
    throw(std::invalid_argument(
      "Could not find part with id:" + std::to_string(_part_id) + " in db."));
  }

  // Find (unique) nodes
  std::vector<std::shared_ptr<Node>> nodes;
  std::vector<size_t> unique_node_indexes;
  std::unordered_set<size_t> unique_node_ids;
  for (size_t iNode = 0; iNode < _node_indexes.size(); ++iNode) {

    // get node
    auto node = db_nodes->get_nodeByIndex(_node_indexes[iNode]);

    // check for duplicate
    auto old_size = unique_node_ids.size();
    unique_node_ids.insert(node->get_nodeID());
    if (unique_node_ids.size() == old_size)
      continue;

    // get node
    nodes.push_back(node);
    unique_node_indexes.push_back(_node_indexes[iNode]);
  }

  // Create element
  auto element =
    create_element_unchecked(_eType, _elementID, unique_node_indexes);

  // Register Element
  for (auto& node : nodes)
    node->add_element(element);
  part->add_element(element);

  return element;
}

/** Add an element to the database from node id
 *
 * @param _etype : type of the element
 * @param _elementID : id of the element
 * @param _part_id : id of the part the element belongs to
 * @param _node_ids : ids of nodes
 */
std::shared_ptr<Element>
DB_Elements::add_elementByNodeID(const Element::ElementType _eType,
                                 int32_t _elementID,
                                 int32_t _part_id,
                                 const std::vector<int32_t>& _node_ids)
{
  if (_elementID < 0) {
    throw(std::invalid_argument("Element-ID may not be negative!"));
  }

  // Find part
  const auto part = db_parts->get_partByID(_part_id);
  if (part == nullptr) {
    throw(std::invalid_argument(
      "Could not find part with id:" + std::to_string(_part_id) + " in db."));
  }

  // Find (unique) nodes
  std::vector<std::shared_ptr<Node>> nodes;
  std::vector<size_t> unique_node_indexes;
  std::unordered_set<size_t> unique_node_ids;
  for (size_t iNode = 0; iNode < _node_ids.size(); ++iNode) {

    // get node (fast)
    size_t node_index = db_nodes->get_index_from_id(_node_ids[iNode]);
    auto node = db_nodes->get_nodeByIndex(node_index);

    // check for duplicate
    auto old_size = unique_node_ids.size();
    unique_node_ids.insert(node->get_nodeID());
    if (unique_node_ids.size() == old_size)
      continue;

    // get node
    nodes.push_back(node);
    unique_node_indexes.push_back(node_index);
  }

  // Create element
  auto element =
    create_element_unchecked(_eType, _elementID, unique_node_indexes);

  // Register Element
  for (auto& node : nodes)
    node->add_element(element);
  part->add_element(element);

  return element;
}

/** Add an element coming from a D3plot file
 *
 * @param ElementType _eType : type of the element to add, enum in Element.hpp
 * @param int32_t _elementID : id of the element to add
 * @param std::vector<int32_t> _elementData : element data from d3plot, node
 * ids and
 * part
 * id
 * @return std::shared_ptr<Element> element : pointer to created instance
 *
 * Add an element to the db by it's ID  and it's nodeIndexes. Throws an
 * exception
 * if one nodeIndex is invalid or if the elementID is already existing.
 */
std::shared_ptr<Element>
DB_Elements::add_element_byD3plot(const Element::ElementType _eType,
                                  const int32_t _elementID,
                                  const std::vector<int32_t>& _elementData)
{
  if (_elementID < 0) {
    throw(std::invalid_argument("Element-ID may not be negative!"));
  }

  // Find part
  // index is decremented once, since ls-dyna starts at 1 (fortran array
  // style)
  const auto part = this->db_parts->get_partByIndex(_elementData.back() - 1);
  if (part == nullptr) {
    throw(std::invalid_argument(
      "Could not find part with index:" + std::to_string(_elementData.back()) +
      " in db."));
  }

  // Find nodes
  std::set<int32_t> node_ids; // just for testing
  std::vector<std::shared_ptr<Node>> nodes;
  std::vector<size_t> node_indexes;
  for (size_t iNode = 0; iNode < _elementData.size() - 1;
       iNode++) { // last is mat

    // dyna starts at index 1 (fortran), this program at 0 of course
    auto _node = this->db_nodes->get_nodeByIndex(_elementData[iNode] - 1);

    // check if duplicate
    auto tmp = node_ids.size();
    node_ids.insert(_elementData[iNode]);
    if (node_ids.size() == tmp)
      continue;

    // add new node data
    nodes.push_back(_node);
    node_indexes.push_back(_elementData[iNode] - 1);
  }

  // Create element
  auto element = create_element_unchecked(_eType, _elementID, node_indexes);

  // Register Elements
  for (auto& node : nodes) {
    node->add_element(element);
  }
  part->add_element(element);

  return element;
}

/** Add an element coming from a KeyFile/Dyna Input File
 *
 * @param Element::ElementType _eType : type of the element to add, enum in
 * Element.hpp
 * @param int32_t _elementID : id of the element to add
 * @param int32_t part_id : id of the part, the element belongs to
 * @param std::vector<int32_t> _node_ids : node ids of the used nodes
 * @return std::shared_ptr<Element> element : pointer to created instance
 *
 * Add an element to the db by it's ID  and it's nodeIDs. Throws an exception
 * if one nodeID is invalid or if the elementID is already existing. Since a
 * KeyFile may have some weird order, missing parts and nodes are created.
 */
std::shared_ptr<Element>
DB_Elements::add_element_byKeyFile(Element::ElementType _eType,
                                   int32_t _elementID,
                                   int32_t _partid,
                                   const std::vector<int32_t>& _node_ids)
{
  if (_elementID < 0) {
    throw(std::invalid_argument("Element-ID may not be negative!"));
  }

  // Find part (inefficient)
  std::shared_ptr<Part> part = nullptr;
  try {
    part = this->db_parts->get_partByID(_partid);
  } catch (std::invalid_argument) {
    part = this->db_parts->add_partByID(_partid);
  }

  // Find nodes
  std::set<int32_t> node_ids;
  std::vector<std::shared_ptr<Node>> nodes;
  std::vector<size_t> node_indexes;
  for (size_t iNode = 0; iNode < _node_ids.size(); ++iNode) {

    auto node_index = db_nodes->get_index_from_id(_node_ids[iNode]);
    auto _node = db_nodes->get_nodeByIndex(node_index);

    // check node existance
    /*
    if (_node == nullptr)
      _node = db_nodes->add_node_byKeyFile(_node_ids[iNode], 0., 0., 0.);
    */

    // check if duplicate
    auto tmp = node_ids.size();
    node_ids.insert(_node_ids[iNode]);
    if (node_ids.size() == tmp)
      continue;

    // save node data
    nodes.push_back(_node);
    node_indexes.push_back(node_index);
  }

  // Create element
  auto element = create_element_unchecked(_eType, _elementID, node_indexes);

  // Register Elements
  for (auto& node : nodes) {
    node->add_element(element);
  }
  part->add_element(element);

  return element;
}

/** Get the DynaInputFile pointer
 * @return DnyaInputFile* keyfile
 */
FEMFile*
DB_Elements::get_femfile()
{
  return this->femfile;
}

/** Get the node-db.
 * @return DB_Nodes* db_nodes
 */
DB_Nodes*
DB_Elements::get_db_nodes()
{
  return this->db_nodes;
}

/** Reserve memory for future elements
 * @param _type element type to apply reserve on
 * @param _size size to reserve internally
 *
 * Does nothing if _type is NONE.
 */
void
DB_Elements::reserve(const Element::ElementType _type, const size_t _size)
{
  if (_type == Element::BEAM) {
    elements2.reserve(_size);
  } else if (_type == Element::SHELL) {
    elements4.reserve(_size);
  } else if (_type == Element::SOLID) {
    elements8.reserve(_size);
  } else if (_type == Element::TSHELL) {
    elements4th.reserve(_size);
  } else {
    throw std::invalid_argument(
      "Can not reserve memory for an unknown ElementType: " +
      std::to_string(_type));
  }
}

/** Get the number of  in the db.
 * @return unsigned int32_t nElements : returns the total number of elements
 * in the db
 */
size_t
DB_Elements::get_nElements(const Element::ElementType _type) const
{
  if (_type == Element::BEAM) {
    return elements2.size();
  } else if (_type == Element::SHELL) {
    return elements4.size();
  } else if (_type == Element::SOLID) {
    return elements8.size();
  } else if (_type == Element::TSHELL) {
    return elements4th.size();
  }
  return elements4.size() + elements2.size() + elements8.size() +
         elements4th.size();
}

/** Get the elements of the database of a certain type
 *
 * @param _type : optional filtering type
 * @return elems : std::vector of elements
 */
std::vector<std::shared_ptr<Element>>
DB_Elements::get_elements(const Element::ElementType _type)
{

  if (_type == Element::NONE) {
    std::vector<std::shared_ptr<Element>> elems;
    elems.reserve(get_nElements(_type));
    elems.insert(elems.end(), elements2.begin(), elements2.end());
    elems.insert(elems.end(), elements4.begin(), elements4.end());
    elems.insert(elems.end(), elements8.begin(), elements8.end());
    elems.insert(elems.end(), elements4th.begin(), elements4th.end());
    return std::move(elems);
  } else if (_type == Element::BEAM) {
    return elements2;
  } else if (_type == Element::SHELL) {
    return elements4;
  } else if (_type == Element::SOLID) {
    return elements8;
  } else if (_type == Element::TSHELL) {
    return elements4th;
  }

  throw(std::invalid_argument("Unknown element type specified."));
}

} // namespace qd