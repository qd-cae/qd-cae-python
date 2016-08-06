
#include "FEMFile.hpp"
#include "../db/DB_Nodes.h"
#include "../db/DB_Parts.h"
#include "../db/DB_Elements.h"

/** Constructor for a new FEMFile
 */
FEMFile::FEMFile(){
   // assign vars
   this->init_vars();
}


/** Constructor for a new FEMFile from a filepath
 * @param string _filepath
 */
FEMFile::FEMFile(string _filepath){
   // assign vars
   this->init_vars();
   this->filepath = _filepath;
}


/** Destructor for a new FEMFile
 */
FEMFile::~FEMFile(){
   if(this->db_nodes != NULL){
      delete this->db_nodes;
      this->db_nodes = NULL;
   }
   if(this->db_parts != NULL){
      delete this->db_parts;
      this->db_parts = NULL;
   }
   if(this->db_elements != NULL){
      delete this->db_elements;
      this->db_elements = NULL;
   }
}


/** Default initialization of the variables
 */
void FEMFile::init_vars(){
   this->db_nodes = new DB_Nodes(this);
   this->db_parts = new DB_Parts();
   this->db_elements = new DB_Elements(this);
   this->db_nodes->set_db_elements(this->db_elements);
}


/** Reset the currents file filepath
 * @param string _filepath : new filepath
 */
void FEMFile::set_filepath(string _filepath){
   this->filepath = _filepath;
}

/** Get the currents file filepath
 * @return string filepath
 */
string FEMFile::get_filepath(){
   return this->filepath;
}

/** Return the pointer to the node db.
 * @return DB_Nodes* db_nodes
 */
DB_Nodes* FEMFile::get_db_nodes(){
   return this->db_nodes;
}

/** Return the pointer to the part db.
 * @return DB_Parts* db_parts
 */
DB_Parts* FEMFile::get_db_parts(){
   return this->db_parts;
}

/** Return the pointer to the element db.
 * @return DB_Elements* db_elements
 */
DB_Elements* FEMFile::get_db_elements(){
   return this->db_elements;
}
