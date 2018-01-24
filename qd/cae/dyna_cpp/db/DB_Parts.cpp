
#include "dyna_cpp/db/DB_Parts.hpp"
#include "dyna_cpp/db/FEMFile.hpp"
#include "dyna_cpp/db/Part.hpp"

namespace qd {

/**
 * Constructor
 */
DB_Parts::DB_Parts(FEMFile* _femfile)
  : femfile(_femfile)
{}

/**
 * Destructor
 */
DB_Parts::~DB_Parts()
{
#ifdef QD_DEBUG
  std::cout << "DB_Parts::~DB_Parts called." << std::endl;
#endif
}

/** Create a part with it's id. The index is just size + 1.
 */
std::shared_ptr<Part>
DB_Parts::add_partByID(int32_t _partID, const std::string& name)
{
  const auto& it = id2index_parts.find(_partID);
  if (it != id2index_parts.end())
    throw(std::invalid_argument("Trying to insert a part with same id " +
                                std::to_string(_partID) +
                                " twice into the part-db!"));

  auto part = std::make_shared<Part>(_partID, name, this->femfile);
  this->parts.push_back(part);
  this->id2index_parts.insert(
    std::pair<int32_t, size_t>(_partID, this->parts.size() - 1));
  return part;
}

/**
 * Get the parts in the db in a vector.
 */
std::vector<std::shared_ptr<Part>>
DB_Parts::get_parts()
{
  return this->parts;
}

/**
 * Get a part by it's name.
 */
std::shared_ptr<Part>
DB_Parts::get_partByName(const std::string& _name)
{
  for (const auto& part_ptr : parts) {
    if (part_ptr->get_name().compare(_name) == 0) {
      return part_ptr;
    }
  }

  throw(std::invalid_argument("part with name <" + _name + "> does not exist"));
}

/**
 * Get the number of parts in the database.
 */
size_t
DB_Parts::get_nParts() const
{
#ifdef QD_DEBUG
  if (parts.size() != id2index_parts.size())
    throw(std::runtime_error("Part Map and Index-Vector have unequal sizes."));
#endif

  return parts.size();
}

/**
 * Print the parts in the db.
 */
void
DB_Parts::print_parts() const
{
  for (const auto& part_ptr : parts) {
    std::cout << "partID:" << part_ptr->get_partID()
              << " name:" << part_ptr->get_name()
              << " nElems:" << part_ptr->get_elements().size() << "\n";
    std::cout << std::flush;
  }
}

} // namespace qd