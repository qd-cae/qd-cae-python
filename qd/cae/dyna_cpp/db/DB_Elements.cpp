
#include <string>
#include <vector>

#include "DB_Elements.hpp"
#include "DB_Nodes.hpp"
#include "DB_Parts.hpp"
#include "Element.hpp"
#include "FEMFile.hpp"
#include "Node.hpp"
#include "Part.hpp"

using namespace std;

/** Constructor
 *
 * @param FEMFile* _femfile : parent file
 */
DB_Elements::DB_Elements(FEMFile *_femfile) {

  this->femfile = _femfile;
  this->db_nodes = _femfile->get_db_nodes();
  this->db_parts = _femfile->get_db_parts();
}

/*
 * Destructor.
 */
DB_Elements::~DB_Elements() {}

/** Add an element coming from a D3plot file
 *
 * @param ElementType _eType : type of the element to add, enum in Element.hpp
 * @param int _elementID : id of the element to add
 * @param vector<int> _elementData : element data from d3plot, node ids and part
 * id
 * @return Element* element : pointer to created instance
 *
 * Add an element to the db by it's ID  and it's nodeIndexes. Throws an
 * exception
 * if one nodeIndex is invalid or if the elementID is already existing.
 */
Element *DB_Elements::add_element_byD3plot(const ElementType _eType,
                                           const int _elementID,
                                           const vector<int> &_elementData) {

  if (_elementID < 0) {
    throw(string("Element-ID may not be negative!"));
  }

  // Find part
  // index is decremented once, since ls-dyna starts at 1 (fortran array style)
  Part *part = this->db_parts->get_part_byIndex(_elementData.back() - 1);
  if (part == nullptr) {
    throw(string("Could not find part with index:") +
          to_string(_elementData.back()) + string(" in db."));
  }

  // Find nodes
  vector<Node *> nodes;
  vector<size_t> node_indexes;
  for (size_t iNode = 0; iNode < _elementData.size() - 1;
       iNode++) { // last is mat
    Node *_node = this->db_nodes->get_nodeByIndex(
        _elementData[iNode] -
        1); // dyna starts at index 1, this program at 0 of course
    if (_node == nullptr)
      throw(string("A node with index:") + to_string(_elementData[iNode]) +
            string(" does not exist and can not be added to an element."));
    if (iNode > 0 && (_elementData[iNode] == _elementData[iNode - 1]))
      break; // repeating an id means that there are just dummy ids

    nodes.push_back(_node);
    node_indexes.push_back(_elementData[iNode] - 1);
  }

  // Create element
  Element *element_raw_ptr;
  unique_ptr<Element> element(
      new Element(_elementID, _eType, node_indexes, this));

  if (_eType == BEAM) {

    unordered_map<int, size_t>::iterator it =
        this->id2index_elements2.find(_elementID);
    if (it != this->id2index_elements2.end()) {
      throw(string("Trying to insert an element with same id twice:") +
            to_string(_elementID));
    }

    this->id2index_elements2.insert(
        pair<int, size_t>(_elementID, this->elements2.size()));
    this->elements2.push_back(std::move(element));
    element_raw_ptr = this->elements2.back().get();

  } else if (_eType == SHELL) {

    unordered_map<int, size_t>::iterator it =
        this->id2index_elements4.find(_elementID);
    if (it != this->id2index_elements4.end()) {
      throw(string("Trying to insert an element with same id twice:") +
            to_string(_elementID));
    }

    this->id2index_elements4.insert(
        pair<int, size_t>(_elementID, this->elements4.size()));
    this->elements4.push_back(std::move(element));
    element_raw_ptr = this->elements4.back().get();

  } else if (_eType == SOLID) {

    unordered_map<int, size_t>::iterator it =
        this->id2index_elements8.find(_elementID);
    if (it != this->id2index_elements8.end()) {
      throw(string("Trying to insert an element with same id twice:") +
            to_string(_elementID));
    }

    this->id2index_elements8.insert(
        pair<int, size_t>(_elementID, this->elements8.size()));
    this->elements8.push_back(std::move(element));
    element_raw_ptr = this->elements8.back().get();

  } else {

    throw(string("Element with unknown element type was tried to get inserted "
                 "into the database."));
  }

  // Register Elements
  for (vector<Node *>::iterator it = nodes.begin(); it != nodes.end(); ++it) {
    ((Node *)*it)->add_element(element_raw_ptr);
  }
  part->add_element(element_raw_ptr);

  return element_raw_ptr;
}

/** Add an element coming from a KeyFile/Dyna Input File
 *
 * @param ElementType _eType : type of the element to add, enum in Element.hpp
 * @param int _elementID : id of the element to add
 * @param int part_id : id of the part, the element belongs to
 * @param vector<int> _node_ids : node ids of the used nodes
 * @return Element* element : pointer to created instance
 *
 * Add an element to the db by it's ID  and it's nodeIDs. Throws an exception
 * if one nodeID is invalid or if the elementID is already existing. Since a
 * KeyFile may have some weird order, missing parts and nodes are created.
 */
Element *DB_Elements::add_element_byKeyFile(ElementType _eType, int _elementID,
                                            int _partid,
                                            vector<int> _node_ids) {
  if (_elementID < 0) {
    throw(string("Element-ID may not be negative!"));
  }

  // Find part
  Part *part = this->db_parts->get_part_byID(_partid);
  if (part == nullptr) {
    part = this->db_parts->add_part_byID(_partid);
  }

  // Find nodes
  vector<Node *> nodes;
  vector<size_t> node_indexes;
  for (size_t iNode = 0; iNode < _node_ids.size(); ++iNode) {
    Node *_node = this->db_nodes->get_nodeByID(_node_ids[iNode]);
    if (_node == nullptr)
      _node =
          this->db_nodes->add_node(_node_ids[iNode], vector<float>(3, 0.0f));
    if (iNode > 0)
      if (_node_ids[iNode] == _node_ids[iNode - 1])
        break; // dummy ids start
    nodes.push_back(_node);
    node_indexes.push_back(this->db_nodes->get_index_from_id(_node_ids[iNode]));
  }

  // Create element
  Element *element_raw_ptr;
  unique_ptr<Element> element(
      new Element(_elementID, _eType, node_indexes, this));

  if (_eType == BEAM) {

    unordered_map<int, size_t>::iterator it =
        this->id2index_elements2.find(_elementID);
    if (it != this->id2index_elements2.end()) {
      throw(string("Trying to insert an element with same id twice:") +
            to_string(_elementID));
    }

    this->id2index_elements2.insert(
        pair<int, size_t>(_elementID, this->elements2.size()));
    this->elements2.push_back(std::move(element));
    element_raw_ptr = this->elements2.back().get();

  } else if (_eType == SHELL) {

    unordered_map<int, size_t>::iterator it =
        this->id2index_elements4.find(_elementID);
    if (it != this->id2index_elements4.end()) {
      throw(string("Trying to insert an element with same id twice:") +
            to_string(_elementID));
    }

    this->id2index_elements4.insert(
        pair<int, size_t>(_elementID, this->elements4.size()));
    this->elements4.push_back(std::move(element));
    element_raw_ptr = this->elements4.back().get();

  } else if (_eType == SOLID) {

    unordered_map<int, size_t>::iterator it =
        this->id2index_elements8.find(_elementID);
    if (it != this->id2index_elements8.end()) {
      throw(string("Trying to insert an element with same id twice:") +
            to_string(_elementID));
    }

    this->id2index_elements8.insert(
        pair<int, size_t>(_elementID, this->elements8.size()));
    this->elements8.push_back(std::move(element));
    element_raw_ptr = this->elements8.back().get();

  } else {

    throw(string("Element with unknown element type was tried to get inserted "
                 "into the database."));
  }

  // Register Elements
  // for(auto node : nodes) {
  for (vector<Node *>::iterator it = nodes.begin(); it != nodes.end(); it++) {
    ((Node *)*it)->add_element(element_raw_ptr);
  }
  part->add_element(element_raw_ptr);

  return element_raw_ptr;
}

/** Get the DynaInputFile pointer
 * @return DnyaInputFile* keyfile
 */
FEMFile *DB_Elements::get_femfile() { return this->femfile; }

/** Get the node-db.
 * @return DB_Nodes* db_nodes
 */
DB_Nodes *DB_Elements::get_db_nodes() { return this->db_nodes; }

/** Reserve memory for future elements
 * @param _type element type to apply reserve on
 * @param _size size to reserve internally
 *
 * Does nothing if _type is NONE.
 */
void DB_Elements::reserve(const ElementType _type, const size_t _size) {

  if (_type == BEAM) {
    elements2.reserve(_size);
  } else if (_type == SHELL) {
    elements4.reserve(_size);
  } else if (_type == SOLID) {
    elements8.reserve(_size);
  }
}

/** Get the number of  in the db.
 * @return unsigned int nElements : returns the total number of elements in the
 * db
 */
size_t DB_Elements::size(ElementType _type) {

  if (_type == BEAM) {
    return elements2.size();
  } else if (_type == SHELL) {
    return elements4.size();
  } else if (_type == SOLID) {
    return elements8.size();
  }
  return elements4.size() + elements2.size() + elements8.size();
}
