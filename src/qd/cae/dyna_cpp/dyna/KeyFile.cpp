
#include <iostream>
#include <fstream>
#include <stdexcept>      // std::invalid_argument
#include "../utility/BoostException.hpp"
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string/trim.hpp>
#include "../utility/FileUtility.hpp"
#include "../db/DB_Nodes.hpp"
#include "../db/DB_Elements.hpp"
#include "../db/DB_Parts.hpp"
#include "../db/Element.hpp"
#include "KeyFile.hpp"

using namespace boost::algorithm;

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
   #ifdef QD_DEBUG
   cout << "Reading teyt file into buffer ... " << flush;
   #endif
   vector<string> lines = FileUtility::read_textFile(_filepath);
   #ifdef QD_DEBUG
   cout << "done." << endl;
   #endif

   // Get databases
   DB_Nodes* db_nodes = this->get_db_nodes();
   DB_Elements* db_elements = this->get_db_elements();

   // Time to do the thing
   bool nodesection = false;
   bool elemsection = false;
   bool elemthicksection = false;
   bool partsection = false;
   bool propsection = false;
   bool propsection_title = false;

   string line;
   vector<float> coords(3);
   vector<int> elemNodes_shell(4);
   int id, partID;
   #ifdef QD_DEBUG
   cout << "Parsing Text File ... " << endl;
   #endif
   for(vector<string>::size_type iLine = 0; iLine != lines.size(); iLine++) {

      line = lines[iLine];

      // TODO Line preprocessing
      // Remove comment stuff behind "$"

      /* NODES */
      if(trim_copy(line) == "*NODE"){
         nodesection = true;
         #ifdef QD_DEBUG
         cout << "Starting *NODE in line: " << iLine << endl;
         #endif
      } else if(nodesection & (line.find('$') == string::npos)
                            & (line.find('*') == string::npos) ){

         try {
            //vector<float> coords(3);
            /*
            cout << "line.substr(0,8) |" << boost::algorithm::trim_copy(line.substr(0,8)) << "|" << endl;
            boost::lexical_cast<int>(boost::algorithm::trim_copy(line.substr(0,8)));
            cout << "line.substr(8,25) |" << boost::algorithm::trim_copy(line.substr(8,16)) << "|" << endl;
            boost::lexical_cast<float>(boost::algorithm::trim_copy(line.substr(8,16)));
            cout << "line.substr(25,40) |" << boost::algorithm::trim_copy(line.substr(24,16)) << "|" << endl;
            boost::lexical_cast<float>(boost::algorithm::trim_copy(line.substr(24,16)));
            cout << "line.substr(40,56) |" << boost::algorithm::trim_copy(line.substr(40,16)) << "|" << endl;
            boost::lexical_cast<float>(boost::algorithm::trim_copy(line.substr(40,16)));
            */
            coords[0] = boost::lexical_cast<float>(trim_copy(line.substr(8,16)));
            coords[1] = boost::lexical_cast<float>(trim_copy(line.substr(24,16)));
            coords[2] = boost::lexical_cast<float>(trim_copy(line.substr(40,16)));
            db_nodes->add_node(boost::lexical_cast<int>(trim_copy(line.substr(0,8))),coords);
         } catch (const std::exception& ex){
            cerr << "Error reading node in line " << iLine << ":" << ex.what() << endl;
            return;
         } catch (const string& ex) {
            cerr << "Error reading node in line " << iLine << ":" << ex << endl;
            return;
         } catch (...) {
            cerr << "Error reading node in line " << iLine << ": Unknown error." << endl;
         }
      } else if( nodesection & ((line.find('*') != string::npos) | line.empty()) ){
         nodesection = false;
         #ifdef QD_DEBUG
         cout << "*NODE finished in line: " << iLine << endl;
         #endif
      }


      /* ELEMENTS */
      if(trim_copy(line) == "*ELEMENT_SHELL"){
         elemsection = true;
         #ifdef QD_DEBUG
         cout << "Starting *ELEMENT_SHELL in line: " << iLine << endl;
         #endif
      } else if(elemsection & (line.find('$') == string::npos)
                            & (line.find('*') == string::npos) ){

         try {
            id = boost::lexical_cast<int>(trim_copy(line.substr(0,8)));
            partID = boost::lexical_cast<int>(trim_copy(line.substr(8,8)));
            elemNodes_shell[0] = boost::lexical_cast<int>(trim_copy(line.substr(16,8)));
            elemNodes_shell[1] = boost::lexical_cast<int>(trim_copy(line.substr(24,8)));
            elemNodes_shell[2] = boost::lexical_cast<int>(trim_copy(line.substr(32,8)));
            elemNodes_shell[3] = boost::lexical_cast<int>(trim_copy(line.substr(40,8)));
            db_elements->add_element_byKeyFile(SHELL, id, partID, elemNodes_shell);
         } catch (const std::exception& ex){
            cerr << "Error reading element in line " << iLine << ":" << ex.what() << endl;
            return;
         } catch (const string& ex) {
            cerr << "Error reading element in line " << iLine << ":" << ex << endl;
            return;
         } catch (...) {
            cerr << "Error reading element in line " << iLine << ": Unknown error." << endl;
         }
      } else if( elemsection &  ((line.find('*') != string::npos) | line.empty()) ){
         elemsection = false;
         #ifdef QD_DEBUG
         cout << "*ELEMENT_SHELL finished in line: " << iLine << endl;
         #endif
      }

   } // for lines
   #ifdef QD_DEBUG
   cout << "parsing of text-file done." << endl;
   #endif


}
