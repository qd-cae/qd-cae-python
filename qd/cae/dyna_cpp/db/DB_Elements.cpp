
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
                                      int32_t _element_id,
                                      int32_t _part_id,
                                      const std::vector<int32_t>& _node_ids)
{
  std::shared_ptr<Element> element =
    std::make_shared<Element>(_element_id, _part_id, _eType, _node_ids, this);

  switch (_eType) {

    case (Element::SHELL):
#pragma omp critical
    {
      // std::lock_guard<std::mutex> lock(_elem4_mutex);

      if (id2index_elements4.find(_element_id) != id2index_elements4.end())
        throw(std::invalid_argument(
          "Trying to insert an element with same id twice:" +
          std::to_string(_element_id)));
      id2index_elements4.insert(
        std::pair<int32_t, size_t>(_element_id, elements4.size()));
      elements4.push_back(element);
    } break;

    case (Element::SOLID):
#pragma omp critical
    {
      // std::lock_guard<std::mutex> lock(_elem8_mutex);

      if (id2index_elements8.find(_element_id) != id2index_elements8.end())
        throw(std::invalid_argument(
          "Trying to insert an element with same id twice:" +
          std::to_string(_element_id)));

      id2index_elements8.insert(
        std::pair<int32_t, size_t>(_element_id, elements8.size()));
      elements8.push_back(element);
    } break;

    case (Element::BEAM):
#pragma omp critical
    {
      // std::lock_guard<std::mutex> lock(_elem2_mutex);

      if (id2index_elements2.find(_element_id) != id2index_elements2.end())
        throw(std::invalid_argument(
          "Trying to insert an element with same id twice:" +
          std::to_string(_element_id)));

      this->id2index_elements2.insert(
        std::pair<int32_t, size_t>(_element_id, elements2.size()));
      this->elements2.push_back(element);
    } break;

    case (Element::TSHELL):
#pragma omp critical
    {
      // std::lock_guard<std::mutex> lock(_elem4th_mutex);

      if (id2index_elements4th.find(_element_id) != id2index_elements4th.end())
        throw(std::invalid_argument(
          "Trying to insert an element with same id twice:" +
          std::to_string(_element_id)));

      id2index_elements4th.insert(
        std::pair<int32_t, size_t>(_element_id, this->elements4th.size()));
      elements4th.push_back(element);
    } break;

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
  std::vector<int32_t> unique_node_ids;
  std::unordered_set<int32_t> node_ids_set;
  for (size_t iNode = 0; iNode < _node_indexes.size(); ++iNode) {

    // get node
    auto node = db_nodes->get_nodeByIndex(_node_indexes[iNode]);

    // check for duplicate
    auto old_size = node_ids_set.size();
    node_ids_set.insert(node->get_nodeID());
    if (node_ids_set.size() == old_size)
      continue;

    // get node
    nodes.push_back(node);
    unique_node_ids.push_back(node->get_nodeID());
  }

  // Create element
  auto element =
    create_element_unchecked(_eType, _part_id, _elementID, unique_node_ids);

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
  std::vector<int32_t> unique_node_ids;
  std::unordered_set<int32_t> node_ids_set;
  for (size_t iNode = 0; iNode < _node_ids.size(); ++iNode) {

    // get node (fast)
    size_t node_index = db_nodes->get_index_from_id(_node_ids[iNode]);
    auto node = db_nodes->get_nodeByIndex(node_index);

    // check for duplicate
    auto old_size = node_ids_set.size();
    node_ids_set.insert(node->get_nodeID());
    if (node_ids_set.size() == old_size)
      continue;

    // get node
    nodes.push_back(node);
    unique_node_ids.push_back(_node_ids[iNode]);
  }

  // Create element
  auto element =
    create_element_unchecked(_eType, _elementID, _part_id, unique_node_ids);

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
  auto part = this->db_parts->get_partByIndex(_elementData.back() - 1);
  if (part == nullptr) {
    // part = this->db_parts->add_partByID(_elementData.back() - 1);
    throw(std::invalid_argument(
      "Could not find part with index:" + std::to_string(_elementData.back()) +
      " in db."));
  }

  // Find nodes
  std::unordered_set<int32_t> node_id_set; // just for testing
  std::vector<int32_t> node_ids;
  std::vector<std::shared_ptr<Node>> nodes;
  for (size_t iNode = 0; iNode < _elementData.size() - 1;
       iNode++) { // last is mat

    // dyna starts at index 1 (fortran), this program at 0 of course
    auto _node = this->db_nodes->get_nodeByIndex(_elementData[iNode] - 1);

    // check if duplicate
    auto tmp = node_id_set.size();
    node_id_set.insert(_elementData[iNode]);
    if (node_id_set.size() == tmp)
      continue;

    // add new node data
    nodes.push_back(_node);
    node_ids.push_back(_node->get_nodeID());
  }

  // Create element
  auto element =
    create_element_unchecked(_eType, _elementID, part->get_partID(), node_ids);

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
  switch (_type) {
    case Element::BEAM:
      elements2.reserve(_size);
      break;
    case Element::SHELL:
      elements4.reserve(_size);
      break;
    case Element::SOLID:
      elements8.reserve(_size);
      break;
    case Element::TSHELL:
      elements4th.reserve(_size);
      break;
    default:
      throw std::invalid_argument(
        "Can not reserve memory for an unknown ElementType: " +
        std::to_string(_type));
      break;
  }
}

/** Get the number of  in the db.
 * @return unsigned int32_t nElements : returns the total number of elements
 * in the db
 */
size_t
DB_Elements::get_nElements(const Element::ElementType _type) const
{
  switch (_type) {
    case Element::BEAM:
      return elements2.size();
    case Element::SHELL:
      return elements4.size();
    case Element::SOLID:
      return elements8.size();
    case Element::TSHELL:
      return elements4th.size();
    case Element::NONE:
      return elements4.size() + elements2.size() + elements8.size() +
             elements4th.size();
  }

  throw(std::invalid_argument("Unknown element type specified."));
}

/** Get the elements of the database of a certain type
 *
 * @param _type : optional filtering type
 * @return elems : std::vector of elements
 */
std::vector<std::shared_ptr<Element>>
DB_Elements::get_elements(const Element::ElementType _type)
{

  switch (_type) {
    case Element::BEAM:
      return elements2;
    case Element::SHELL:
      return elements4;
    case Element::SOLID:
      return elements8;
    case Element::TSHELL:
      return elements4th;
    case Element::NONE: {
      std::vector<std::shared_ptr<Element>> elems;
      elems.reserve(get_nElements(_type));
      elems.insert(elems.end(), elements2.begin(), elements2.end());
      elems.insert(elems.end(), elements4.begin(), elements4.end());
      elems.insert(elems.end(), elements8.begin(), elements8.end());
      elems.insert(elems.end(), elements4th.begin(), elements4th.end());
      return elems;
    }
  }

  throw(std::invalid_argument("Unknown element type specified."));
}

/** Get the element ids
 *
 * @param element_filter : filter type for elements
 * @return tensor
 */
Tensor_ptr<int32_t>
DB_Elements::get_element_ids(Element::ElementType element_filter)
{

  size_t element_offset = 0;
  auto tensor = std::make_shared<Tensor<int32_t>>();
  tensor->resize({ this->get_nElements(element_filter) });
  auto& data = tensor->get_data();

  if (element_filter == Element::NONE || element_filter == Element::BEAM) {
    for (auto& elem : elements2)
      data[element_offset++] = elem->get_elementID();
  }

  if (element_filter == Element::NONE || element_filter == Element::SHELL) {
    for (auto& elem : elements4)
      data[element_offset++] = elem->get_elementID();
  }

  if (element_filter == Element::NONE || element_filter == Element::SOLID) {
    for (auto& elem : elements8)
      data[element_offset++] = elem->get_elementID();
  }

  if (element_filter == Element::NONE || element_filter == Element::TSHELL) {
    for (auto& elem : elements4th)
      data[element_offset++] = elem->get_elementID();
  }

  return tensor;
}

/** Get the node ids of elements
 *
 * @param element_type : type of the elements
 * @param n_nodes : number of nodes
 * @return tensor
 */
Tensor_ptr<int32_t>
DB_Elements::get_element_node_ids(Element::ElementType element_type,
                                  size_t n_nodes)
{

  size_t offset = 0;
  auto tensor = std::make_shared<Tensor<int32_t>>();
  tensor->resize({ this->get_nElements(element_type), n_nodes });
  auto& data = tensor->get_data();

  switch (element_type) {
    case Element::BEAM:
      for (auto& elem : elements2) {
        if (elem->get_nNodes() == n_nodes) {
          const auto& node_ids = elem->get_node_ids();
          std::copy(node_ids.begin(), node_ids.end(), data.begin() + offset);
          offset += node_ids.size();
        }
      }
      break;
    case Element::SHELL:
      for (auto& elem : elements4) {
        if (elem->get_nNodes() == n_nodes) {
          const auto& node_ids = elem->get_node_ids();
          std::copy(node_ids.begin(), node_ids.end(), data.begin() + offset);
          offset += node_ids.size();
        }
      }
      break;
    case Element::SOLID:
      for (auto& elem : elements8) {
        if (elem->get_nNodes() == n_nodes) {
          const auto& node_ids = elem->get_node_ids();
          std::copy(node_ids.begin(), node_ids.end(), data.begin() + offset);
          offset += node_ids.size();
        }
      }
      break;
    case Element::TSHELL:
      for (auto& elem : elements4th) {
        if (elem->get_nNodes() == n_nodes) {
          const auto& node_ids = elem->get_node_ids();
          std::copy(node_ids.begin(), node_ids.end(), data.begin() + offset);
          offset += node_ids.size();
        }
      }
      break;
    case Element::NONE:
    default:
      break;
  }

  return tensor;
}

/** Get the energy of elements
 *
 * @param element_filter : optional element filter
 * @return tensor : result data array
 *
 * If a value is not present for an element, the it will be initialized as 0 by
 * default.
 */
Tensor_ptr<float>
DB_Elements::get_element_energy(Element::ElementType element_filter)
{
  constexpr auto default_value = 0.;
  size_t offset = 0;
  auto tensor = std::make_shared<Tensor<float>>();

  // resize
  const auto nTimesteps = get_femfile()->get_nTimesteps();
  tensor->resize({ this->get_nElements(element_filter), nTimesteps });
  auto& data = tensor->get_data();

  auto elements = get_elements(element_filter);

  for (auto& element : elements) {
    auto result = element->get_energy();
    if (result.size() != 0) {
      std::copy(result.begin(), result.end(), data.begin() + offset);
      offset += nTimesteps;
    } else {
      std::fill(data.begin() + offset,
                data.begin() + offset + nTimesteps,
                default_value);
      offset += nTimesteps;
    }
  }

  return tensor;
}

/** Get the mises stress of elements
 *
 * @param element_filter : optional element filter
 * @return tensor : result data array
 *
 * If a value is not present for an element, the it will be initialized as 0 by
 * default.
 */
Tensor_ptr<float>
DB_Elements::get_element_stress_mises(Element::ElementType element_filter)
{
  constexpr auto default_value = 0.;
  size_t offset = 0;
  auto tensor = std::make_shared<Tensor<float>>();

  // resize
  const auto nTimesteps = get_femfile()->get_nTimesteps();
  tensor->resize({ this->get_nElements(element_filter), nTimesteps });
  auto& data = tensor->get_data();

  auto elements = get_elements(element_filter);
  for (auto& element : elements) {
    auto result = element->get_stress_mises();
    if (result.size() != 0) {
      std::copy(result.begin(), result.end(), data.begin() + offset);
      offset += nTimesteps;
    } else {
      std::fill(data.begin() + offset,
                data.begin() + offset + nTimesteps,
                default_value);
      offset += nTimesteps;
    }
  }

  return tensor;
}

/** Get the plastic strain of elements
 *
 * @param element_filter : optional element filter
 * @return tensor : result data array
 *
 * If a value is not present for an element, the it will be initialized as 0 by
 * default.
 */
Tensor_ptr<float>
DB_Elements::get_element_plastic_strain(Element::ElementType element_filter)
{
  constexpr auto default_value = 0.;
  size_t offset = 0;
  auto tensor = std::make_shared<Tensor<float>>();

  // resize tensor (nElems x nTimesteps)
  const auto nTimesteps = get_femfile()->get_nTimesteps();
  tensor->resize({ this->get_nElements(element_filter), nTimesteps });
  auto& data = tensor->get_data();

  auto elements = get_elements(element_filter);
  for (auto& element : elements) {
    auto result = element->get_plastic_strain();
    if (result.size() != 0) {
      std::copy(result.begin(), result.end(), data.begin() + offset);
      offset += nTimesteps;
    } else {
      std::fill(data.begin() + offset,
                data.begin() + offset + nTimesteps,
                default_value);
      offset += nTimesteps;
    }
  }

  return tensor;
}

/** Get the strain of elements
 *
 * @param element_filter : optional element filter
 * @return tensor : result data array
 *
 * If a value is not present for an element, the it will be initialized as 0 by
 * default.
 */
Tensor_ptr<float>
DB_Elements::get_element_strain(Element::ElementType element_filter)
{
  constexpr auto default_value = 0.;
  constexpr size_t nComponents = 6;

  size_t offset = 0;
  auto tensor = std::make_shared<Tensor<float>>();

  // resize (nElems x nTimesteps x nStrain)
  const auto nTimesteps = get_femfile()->get_nTimesteps();
  tensor->resize(
    { this->get_nElements(element_filter), nTimesteps, nComponents });
  auto& data = tensor->get_data();

  auto elements = get_elements(element_filter);
  for (auto& element : elements) {
    auto result = element->get_strain();

    if (result.size() != 0) {
#ifdef QD_DEBUG
      if (result.size() != nTimesteps)
        throw(std::runtime_error(
          "element timeseries has a wrong number of timesteps."));
#endif

      for (auto& vec : result) {
#ifdef QD_DEBUG
        if (vec.size() != nComponents)
          throw(std::runtime_error("vector has wrong number of components:" +
                                   std::to_string(vec.size()) +
                                   " != " + std::to_string(nComponents)));
#endif
        std::copy(vec.begin(), vec.end(), data.begin() + offset);
        offset += vec.size();
      }

    } else {
      const auto tmp = nTimesteps * nComponents;
      std::fill(
        data.begin() + offset, data.begin() + offset + tmp, default_value);
      offset += tmp;
    }
  }

  return tensor;
}

/** Get the stress of elements
 *
 * @param element_filter : optional element filter
 * @return tensor : result data array
 *
 * If a value is not present for an element, the it will be initialized as 0 by
 * default.
 */
Tensor_ptr<float>
DB_Elements::get_element_stress(Element::ElementType element_filter)
{
  constexpr auto default_value = 0.;
  constexpr size_t nComponents = 6;

  size_t offset = 0;
  auto tensor = std::make_shared<Tensor<float>>();

  // resize (nElems x nTimesteps x nStrain)
  const auto nTimesteps = get_femfile()->get_nTimesteps();
  tensor->resize(
    { this->get_nElements(element_filter), nTimesteps, nComponents });
  auto& data = tensor->get_data();

  const auto element_offset = nTimesteps * nComponents;
  auto elements = get_elements(element_filter);
  for (auto& element : elements) {
    auto result = element->get_stress();

    if (result.size() != 0) {
#ifdef QD_DEBUG
      if (result.size() != nTimesteps)
        throw(std::runtime_error(
          "element timeseries has a wrong number of timesteps."));
#endif

      for (auto& vec : result) {
#ifdef QD_DEBUG
        if (vec.size() != nComponents)
          throw(std::runtime_error("vector has wrong number of components:" +
                                   std::to_string(vec.size()) +
                                   " != " + std::to_string(nComponents)));
#endif
        std::copy(vec.begin(), vec.end(), data.begin() + offset);
        offset += vec.size();
      }

    } else {
      std::fill(data.begin() + offset,
                data.begin() + offset + element_offset,
                default_value);
      offset += element_offset;
    }
  }

  return tensor;
}

/** Get the coords of elements
 *
 * @param element_filter : optional element filter
 * @return tensor : result data array
 *
 * If a value is not present for an element, the it will be initialized as 0 by
 * default.
 */
Tensor_ptr<float>
DB_Elements::get_element_coords(Element::ElementType element_filter)
{
  constexpr auto default_value = 0.;
  constexpr size_t nComponents = 3;

  size_t offset = 0;
  auto tensor = std::make_shared<Tensor<float>>();

  // resize (nElems x nTimesteps x nStrain)
  const auto nTimesteps = get_femfile()->get_nTimesteps();
  tensor->resize(
    { this->get_nElements(element_filter), nTimesteps, nComponents });
  auto& data = tensor->get_data();

  const auto element_offset = nTimesteps * nComponents;
  auto elements = get_elements(element_filter);
  for (auto& element : elements) {
    auto result = element->get_coords();

    if (result.size() != 0) {
#ifdef QD_DEBUG
      if (result.size() != nTimesteps)
        throw(std::runtime_error(
          "element timeseries has a wrong number of timesteps."));
#endif

      for (auto& vec : result) {
#ifdef QD_DEBUG
        if (vec.size() != nComponents)
          throw(std::runtime_error("vector has wrong number of components:" +
                                   std::to_string(vec.size()) +
                                   " != " + std::to_string(nComponents)));
#endif
        std::copy(vec.begin(), vec.end(), data.begin() + offset);
        offset += vec.size();
      }

    } else {
      std::fill(data.begin() + offset,
                data.begin() + offset + element_offset,
                default_value);
      offset += element_offset;
    }
  }

  return tensor;
}

/** Get the history vars for a specific element type
 *
 * @param element_type : type of the element
 * @return tensor : data array
 *
 * In contrast to all other array functions this function actually needs a
 * specific element type for the reason, that history vars usually differ
 * between the element types. We didn't screw this up!
 */
Tensor_ptr<float>
DB_Elements::get_element_history_vars(Element::ElementType element_type)
{

  // test
  if (element_type == Element::ElementType::NONE)
    throw(std::invalid_argument(
      "You need to specify an element type if "
      "requesting history variables for the reason, "
      "that history vars differ between the "
      "element types. Don't blame us, we didn't screw this up!"));

  constexpr auto default_value = 0.;

  size_t offset = 0;
  auto tensor = std::make_shared<Tensor<float>>();

  // get the number of history vars ... little bit complicated
  auto elements = get_elements(element_type);
  const auto nComponents =
    elements.size() != 0 ? elements[0]->get_history_vars().size() : 0;

  // resize (nElems x nTimesteps x nStrain)
  const auto nTimesteps = get_femfile()->get_nTimesteps();
  tensor->resize(
    { this->get_nElements(element_type), nTimesteps, nComponents });
  auto& data = tensor->get_data();

  const auto element_offset = nTimesteps * nComponents;
  for (auto& element : elements) {
    auto result = element->get_history_vars();

    if (result.size() != 0) {
#ifdef QD_DEBUG
      if (result.size() != nTimesteps)
        throw(std::runtime_error(
          "element timeseries has a wrong number of timesteps."));
#endif

      for (auto& vec : result) {
#ifdef QD_DEBUG
        if (vec.size() != nComponents)
          throw(std::runtime_error("vector has wrong number of components:" +
                                   std::to_string(vec.size()) +
                                   " != " + std::to_string(nComponents)));
#endif
        std::copy(vec.begin(), vec.end(), data.begin() + offset);
        offset += vec.size();
      }

    } else {
      std::fill(data.begin() + offset,
                data.begin() + offset + element_offset,
                default_value);
      offset += element_offset;
    }
  }

  return tensor;
}

} // namespace qd