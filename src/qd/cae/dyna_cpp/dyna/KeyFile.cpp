
#include <iostream>
#include <fstream>
#include <stdexcept>      // std::invalid_argument
#include "../utility/BoostException.hpp"
#include <boost/lexical_cast.hpp>
#include "../utility/FileUtility.hpp"
#include "../db/DB_Nodes.hpp"
#include "../db/DB_Elements.hpp"
#include "../db/DB_Parts.hpp"
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

   string line;
   for(vector<string>::size_type iLine = 0; iLine != lines.size(); iLine++) {

      line = lines[iLine];

      /* NODES */
      if(line == "*NODE"){
         nodesection = true;
      } else if(nodesection & (line.find('&') == string::npos)
                            & (line.find('*') == string::npos) ){

         try {
            vector<float> coords(3);
            coords.push_back(boost::lexical_cast<float>(line.substr(9,25)));
            coords.push_back(boost::lexical_cast<float>(line.substr(25,41)));
            coords.push_back(boost::lexical_cast<float>(line.substr(41,57)));
            db_nodes->add_node(boost::lexical_cast<int>(line.substr(0,9)),coords);
         } catch (const std::exception& e){
            cerr << "Error reading node in line " << iLine << ":" << e.what() << endl;
         }

      } else if(line.find('*') != string::npos){
         nodesection = false;
      }

   }

}
