
#ifndef FEMFILE_HPP
#define FEMFILE_HPP

// includes
#include <dyna_cpp/db/DB_Elements.hpp>
#include <dyna_cpp/db/DB_Nodes.hpp>
#include <dyna_cpp/db/DB_Parts.hpp>

#include <string>

// forward declarations
class D3plot;
class KeyFile;
// class DB_Nodes;
// class DB_Parts;
// class DB_Elements;

/** Superclass for all FEM-Files
 */
class FEMFile : public DB_Nodes, public DB_Parts, public DB_Elements {
  // class FEMFile {
 private:
  std::string filepath;
  // std::unique_ptr<DB_Nodes> db_nodes;
  // std::unique_ptr<DB_Parts> db_parts;
  // std::unique_ptr<DB_Elements> db_elements;

 public:
  FEMFile();
  FEMFile(const std::string& filepath);
  virtual ~FEMFile();
  void set_filepath(const std::string& filepath);
  std::string get_filepath();

  inline DB_Nodes* get_db_nodes() { return static_cast<DB_Nodes*>(this); }
  inline DB_Parts* get_db_parts() { return static_cast<DB_Parts*>(this); }
  inline DB_Elements* get_db_elements() {
    return static_cast<DB_Elements*>(this);
  }
  /*
  inline DB_Nodes* get_db_nodes() { return this->db_nodes.get(); }
  inline DB_Parts* get_db_parts() { return this->db_parts.get(); }
  inline DB_Elements* get_db_elements() { return this->db_elements.get(); }
  */
  virtual bool is_d3plot() const = 0;
  virtual bool is_keyFile() const = 0;
  virtual D3plot* get_d3plot() = 0;
  virtual KeyFile* get_keyFile() = 0;
};

#endif
