
#include "dyna_cpp/db/FEMFile.hpp"
#include "dyna_cpp/db/DB_Elements.hpp"
#include "dyna_cpp/db/DB_Nodes.hpp"
#include "dyna_cpp/db/DB_Parts.hpp"

using namespace std;

/** Constructor for a new FEMFile
 */
FEMFile::FEMFile() : DB_Nodes(this), DB_Elements(this), DB_Parts(this) {}

/** Constructor for a new FEMFile from a filepath
 * @param string _filepath
 */
FEMFile::FEMFile(const string& _filepath)
    : DB_Nodes(this), DB_Elements(this), DB_Parts(this), filepath(_filepath) {
  // assign vars
}

/*
FEMFile::FEMFile()
    : db_nodes(std::make_unique<DB_Nodes>(this)),
      db_parts(std::make_unique<DB_Parts>(this)),
      db_elements(std::make_unique<DB_Elements>(this)) {}

FEMFile::FEMFile(const string& _filepath)
    : filepath(_filepath),
      db_nodes(std::make_unique<DB_Nodes>(this)),
      db_parts(std::make_unique<DB_Parts>(this)),
      db_elements(std::make_unique<DB_Elements>(this)) {}
      */

/** Destructor for a new FEMFile
 */
FEMFile::~FEMFile() {
#ifdef QD_DEBUG
  std::cout << "FEMFile::~FEMFile() called." << std::endl;
#endif
}

/** Reset the currents file filepath
 * @param string _filepath : new filepath
 */
void FEMFile::set_filepath(const string& _filepath) {
  this->filepath = _filepath;
}

/** Get the currents file filepath
 * @return string filepath
 */
string FEMFile::get_filepath() { return this->filepath; }
