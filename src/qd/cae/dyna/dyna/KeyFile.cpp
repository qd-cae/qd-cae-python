
#include <iostream>
#include <fstream>
#include "../utility/FileUtility.hpp"
#include "KeyFile.hpp"


/**
 * Constructor for a LS-Dyna input file.
 */
KeyFile::KeyFile() {

}



/** Constructor for reading a LS-Dyna input file.
 *
 * @param string filepath : filepath of a key file to read
 */
KeyFile::KeyFile(string _filepath)
   : FEMFile(_filepath) {

   // Read the mesh
   this->read_mesh(this->get_filepath());

}


/** Read the mesh from the file given in the filepath
 *
 */
void KeyFile::read_mesh(string _filepath){

   // Read the lines
   vector<string> lines = FileUtility::read_textFile(_filepath);

   // Get databases
   DB_Nodes* db_nodes = this->get_db_nodes();

   // Time to do the thing
   bool nodesection = false;
   bool elemsection = false;
   bool elemthicksection = false;
   bool partsection = false;
   bool propsection = false;
   bool propsection_title = false;

   for(vector<string>::size_type iLine = 0; iLine != lines.size(); iLine++) {

      /* NODES */
      /*
      if(lines[iLine] == "*NODE"){
         nodesection = true
      } else if(nodesection & (& not in line) & (* not in line)){
         _nodeID = int(line[0:8])
         _x = float(line[9:24])
         _y = float(line[25:40])
         _z = float(line[41:56])
         db_nodes->add_node(blabla);
      } else if( * in line){
         nodesection = false
      }
      */

   }

}
