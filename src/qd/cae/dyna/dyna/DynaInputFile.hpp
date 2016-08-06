
#ifndef DYNAINPUTFILE_H
#define DYNAINPUTFILE_H

// includes
#include <string>

// forward declarations
class DB_Nodes;
class DB_Parts;
class DB_Elements;

// namespaces
using namespace std;

/**
 * This is a class for reading LS-Dyna input files.The mesh will be parsed with
 * it#s properties, currently only in a limited way.
 */
class DynaInputFile {

private:
   // private vars
   string filepath;
   DB_Nodes* db_nodes;
   DB_Parts* db_parts;
   DB_Elements* db_elements;
   // private functions
   void init_vars();
   void read_mesh();

// === P U B L I C === //
public:
   DynaInputFile();
   DynaInputFile(string filepath);
   ~DynaInputFile();


}

#endif
