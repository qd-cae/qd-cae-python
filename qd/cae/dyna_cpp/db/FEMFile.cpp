
#include "dyna_cpp/db/FEMFile.hpp"
#include "dyna_cpp/db/DB_Elements.hpp"
#include "dyna_cpp/db/DB_Nodes.hpp"
#include "dyna_cpp/db/DB_Parts.hpp"

namespace qd {

/** Constructor for a new FEMFile
 */
FEMFile::FEMFile()
  : DB_Nodes(this)
  , DB_Parts(this)
  , DB_Elements(this)
{
}

/** Constructor for a new FEMFile from a filepath
 * @param std::string _filepath
 */
FEMFile::FEMFile(const std::string& _filepath)
  : DB_Nodes(this)
  , DB_Parts(this)
  , DB_Elements(this)
  , filepath(_filepath)
{
}

/** Destructor for a new FEMFile
 */
FEMFile::~FEMFile()
{
#ifdef QD_DEBUG
  std::cout << "FEMFile::~FEMFile() called." << std::endl;
#endif
}

/** Reset the currents file filepath
 * @param std::string _filepath : new filepath
 */
void
FEMFile::set_filepath(const std::string& _filepath)
{
  this->filepath = _filepath;
}

/** Get the currents file filepath
 * @return std::string filepath
 */
std::string
FEMFile::get_filepath()
{
  return this->filepath;
}

} // namespace qd