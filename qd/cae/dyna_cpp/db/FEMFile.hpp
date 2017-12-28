
#ifndef FEMFILE_HPP
#define FEMFILE_HPP

// includes
#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>

#include <string>

namespace qd {

// forward declarations
class D3plot;
class KeyFile;

/** Superclass for all FEM-Files
 */
class FEMFile
  : public DB_Nodes
  , public DB_Parts
  , public DB_Elements
{

private:
  std::string filepath;

public:
  explicit FEMFile();
  explicit FEMFile(const std::string& filepath);
  virtual ~FEMFile();
  void set_filepath(const std::string& filepath);
  std::string get_filepath();

  inline DB_Nodes* get_db_nodes() { return static_cast<DB_Nodes*>(this); }
  inline DB_Parts* get_db_parts() { return static_cast<DB_Parts*>(this); }
  inline DB_Elements* get_db_elements()
  {
    return static_cast<DB_Elements*>(this);
  }
};

} // namespace qd

#endif
