
#include <iostream>
#include <fstream>
#include "../utility/FileUtility.h"
#include "DynaInputFile.hpp"
#include "../db/DB_Elements.h"
#include "../db/DB_Nodes.h"
#include "../db/DB_Parts.h"

/**
 * Constructor for a LS-Dyna input file.
 */
DynaInputFile::DynaInputFile(){
   this->init_vars();
}


/** Constructor for reading a LS-Dyna input file.
 *
 * @param string filepath : filepath of a key file to read
 */
DynaInputFile::DynaInputFile(string filepath){

   // Assign vars
   this->init_vars();
   this->filepath = filepath;

   // Read the mesh
   this->read_mesh();

}

/**
 * Destructor for the KeyFile
 */
DynaInputFile::~DynaInputFile(){
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

/** Initialize the variables with standard settings
 *
 */
DynaInputFile::init_vars(){

}

/** Read the mesh from a given input file
 *
 */
void DynaInputFile::read_mesh(){

   // Read the lines
   vector<string> lines = FileUtility::read_textFile(this->filepath);

   // Time to do the thing
   for(std::vector<string>::size_type iLine = 0; iLine != v.size(); iLine++) {



   }
