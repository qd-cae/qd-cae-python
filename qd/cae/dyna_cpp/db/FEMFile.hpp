
#ifndef FEMFILE_HPP
#define FEMFILE_HPP

// includes
#include <string>

// forward declarations
class D3plot;
class KeyFile;
class DB_Nodes;
class DB_Parts;
class DB_Elements;

// namespaces
using namespace std;

/** Superclass for all FEM-Files
 */
class FEMFile {

private:
   string filepath;
   DB_Nodes* db_nodes;
   DB_Parts* db_parts;
   DB_Elements* db_elements;
   void init_vars();

public:
   FEMFile();
   FEMFile(string filepath);
   virtual ~FEMFile();
   void set_filepath(string filepath);
   string get_filepath();
   DB_Nodes* get_db_nodes();
   DB_Parts* get_db_parts();
   DB_Elements* get_db_elements();
   virtual bool is_d3plot() = 0;
   virtual bool is_keyFile() = 0;
   virtual D3plot* get_d3plot() = 0;
   virtual KeyFile* get_keyFile() = 0;

};

#endif
