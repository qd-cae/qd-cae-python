
#include "FEMFile.hpp"
#include "../db/DB_Nodes.hpp"
#include "../db/DB_Parts.hpp"
#include "../db/DB_Elements.hpp"

using namespace std;

/** Constructor for a new FEMFile
 */
FEMFile::FEMFile(){
   // assign vars
   this->init_vars();
}


/** Constructor for a new FEMFile from a filepath
 * @param string _filepath
 */
FEMFile::FEMFile(const string& _filepath){
   // assign vars
   this->init_vars();
   this->filepath = _filepath;
}


/** Destructor for a new FEMFile
 */
FEMFile::~FEMFile(){
   if(this->db_nodes != nullptr){
      delete this->db_nodes;
   }
   if(this->db_parts != nullptr){
      delete this->db_parts;
   }
   if(this->db_elements != nullptr){
      delete this->db_elements;
   }
}


/** Default initialization of the variables
 */
void FEMFile::init_vars(){
   this->db_nodes = new DB_Nodes(this);
   this->db_parts = new DB_Parts(this);
   this->db_elements = new DB_Elements(this);
   this->db_nodes->set_db_elements(this->db_elements);
}


/** Reset the currents file filepath
 * @param string _filepath : new filepath
 */
void FEMFile::set_filepath(const string& _filepath){
   this->filepath = _filepath;
}

/** Get the currents file filepath
 * @return string filepath
 */
string FEMFile::get_filepath(){
   return this->filepath;
}

